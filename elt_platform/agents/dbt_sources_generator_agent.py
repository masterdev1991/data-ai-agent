"""
DBT Sources Generator Agent
Automatically generates dbt sources.yml files after data ingestion
"""

# import logging
import os
# from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
import yaml
# from agents.base import BaseAgent
from psycopg2.extras import RealDictCursor
# from utils.file_ops import save_yaml_file
# from utils.logger import get_logger

from elt_platform.core.settings import get_settings

# logger = get_logger(__name__)
settings = get_settings()


class DBTSourcesGenerator:
    """
    DBT Sources Generator Agent to automatically create a dbt source.yml after Ingestion Agent
    """

    def __init__(self, db_url: str):
        # Parse database URL to get connection parameters
        self.db_config = self._parse_db_url(db_url)

    def _parse_db_url(self, db_url: str = settings.DATABASE_URL) -> Dict[str, Any]:
        """Parse PostgreSQL URL into connection parameters"""
        try:
            # Handle postgresql:// URLs
            if db_url.startswith("postgresql://"):
                from urllib.parse import urlparse

                result = urlparse(db_url)
                return {
                    "host": result.hostname,
                    "port": result.port or 5432,
                    "database": result.path[1:] if result.path else "postgres",
                    "user": result.username,
                    "password": result.password,
                }
            else:
                # Fallback for simple connection strings
                return {
                    "host": "localhost",
                    "port": 5432,
                    "database": "etl_db",
                    "user": "user",
                    "password": "password",
                }
        except Exception:
            # Default fallback
            return {
                "host": "localhost",
                "port": 5432,
                "database": "etl_db",
                "user": "user",
                "password": "password",
            }

    def introspect_table_schema(
        self, schema_name: str, table_name: str
    ) -> Dict[str, Any]:
        """
        Introspect Postgres table to get detailed schema information
        """
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Get table columns
                    cursor.execute(
                        """
                        SELECT
                            column_name,
                            data_type,
                            character_maximum_length,
                            is_nullable,
                            column_default,
                            ordinal_position
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (schema_name, table_name),
                    )
                    columns = cursor.fetchall()

                    # Get row count
                    cursor.execute(f"""
                        SELECT COUNT(*) as row_count
                        FROM {schema_name}.{table_name}
                    """)
                    row_count = cursor.fetchone()["row_count"]

                    return {
                        "table_name": table_name,
                        "schema_name": schema_name,
                        "columns": [dict(col) for col in columns],
                        "row_count": row_count,
                    }

        except Exception as e:
            raise Exception(
                f"Error introspecting table {schema_name}.{table_name}: {str(e)}"
            )

    def generate_sources_yaml(
        self,
        table_name: str,
        schema_name: str = "public",
        source_file_path: Optional[str] = None,
    ) -> str:
        """
        Generate dbt sources.yml content
        """

        # Get table schema
        table_schema = self.introspect_table_schema(schema_name, table_name)

        # Generate column configurations
        columns = []
        for col in table_schema["columns"]:
            column_config = {
                "name": col["column_name"],
                "description": self._generate_column_description(col),
            }

            # Add tests for important columns
            tests = self._determine_column_tests(col)
            if tests:
                column_config["tests"] = tests

            columns.append(column_config)

        # Build table configuration
        table_config = {
            "name": table_name,
            "description": f"Raw data table containing {table_schema['row_count']} rows",
            "columns": columns,
        }

        # Build source configuration
        source_config = {
            "name": f"{schema_name}_raw",
            "description": f"Raw data source for {schema_name} schema",
            "database": self.db_config["database"],
            "schema": schema_name,
            "tables": [table_config],
        }

        sources_config = {
            "version": 2,
            "sources": [source_config],
        }

        return yaml.dump(sources_config, default_flow_style=False, sort_keys=False)

    def _generate_column_description(self, col: Dict[str, Any]) -> str:
        """
        Generate intelligent column descriptions
        """
        column_name = col["column_name"].lower()
        data_type = col["data_type"].lower()

        if "id" in column_name and column_name.endswith("id"):
            if column_name == "id":
                return "Primary identifier for this record"
            else:
                entity = column_name.replace("_id", "").replace("id", "")
                return f"Foreign key reference to {entity}"
        elif "name" in column_name:
            return f"Name or title field ({data_type})"
        elif "email" in column_name:
            return "Email address"
        elif "date" in column_name or "time" in column_name:
            return f"Date/time field indicating {column_name.replace('_', ' ')}"
        elif "created" in column_name:
            return "Record creation timestamp"
        elif "updated" in column_name:
            return "Record last update timestamp"
        elif "status" in column_name:
            return "Status indicator"
        elif "amount" in column_name or "price" in column_name:
            return f"Monetary amount ({data_type})"
        else:
            return f"Column {col['column_name']} of type {data_type}"

    def _determine_column_tests(self, col: Dict[str, Any]) -> List[str]:
        """
        Determine appropriate dbt tests for each column
        """
        tests = []
        column_name = col["column_name"].lower()

        # Not null test for non-nullable columns
        if col["is_nullable"] == "NO":
            tests.append("not_null")

        # Unique test for ID columns
        if column_name == "id" or (column_name.endswith("_id") and "id" in column_name):
            tests.append("unique")

        return tests

    def save_sources_file(
        self, sources_yaml: str, table_name: str, output_dir: str = "./dbt_sources"
    ) -> str:
        """
        Save the generated sources.yml file
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"src_{table_name}.yml"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "w") as f:
            f.write(sources_yaml)

        return file_path
