# ingestion.py

import os
import json
from io import StringIO
import pandas as pd
import requests
from sqlalchemy import create_engine, BigInteger, Text, JSON, Float, Boolean, DateTime
from elt_platform.core.settings import get_settings

settings = get_settings()


class IngestionAgent:
    """
    Ingestion Agent for loading the data to the database
    """

    def __init__(self, default_db_url: str = settings.DATABASE_URL):
        self.current_df = None
        self.history = []
        self.default_db_url = default_db_url

    def extract_csv(self, filename: str) -> str:
        try:
            if filename.startswith(("http://", "https://")):
                return self._extract_from_url(filename)
            return self._extract_from_file(filename)
        except Exception as e:
            return f"Error: {str(e)}"

    def _extract_from_file(self, filename: str) -> str:
        filepath = os.path.join(os.getcwd(), filename)
        if not os.path.exists(filepath):
            return f"File not found: {filepath}"
        self.current_df = pd.read_csv(filepath)
        self.history.append(f"Extracted {filename}")
        return f"Loaded {len(self.current_df)} rows from {filename}\nColumns: {list(self.current_df.columns)}"

    def _extract_from_url(self, url: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,application/csv,text/plain,*/*",
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()

        if "json" in content_type or url.endswith(".json"):
            data = response.json()
            if isinstance(data, list):
                self.current_df = pd.DataFrame(data)
            elif isinstance(data, dict):
                for key in ["data", "results", "items", "records", "rows"]:
                    if key in data and isinstance(data[key], list):
                        self.current_df = pd.DataFrame(data[key])
                        break
                else:
                    self.current_df = pd.json_normalize(data)
            else:
                return f"Unsupported JSON structure from {url}"
        else:
            self.current_df = pd.read_csv(StringIO(response.text))

        self.history.append(f"Extracted from API: {url}")
        return f"Loaded {len(self.current_df)} rows from API: {url}\nColumns: {list(self.current_df.columns)}"

    def extract_api(self, url: str, headers: str = None) -> str:
        try:
            request_headers = {
                "User-Agent": "ETL-Agent/1.0",
                "Accept": "application/json,text/csv,*/*",
            }
            if headers:
                for header_pair in headers.split(","):
                    if ":" in header_pair:
                        key, value = header_pair.split(":", 1)
                        request_headers[key.strip()] = value.strip()

            response = requests.get(url, headers=request_headers, timeout=30)
            response.raise_for_status()
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
                if isinstance(data, list):
                    self.current_df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    for key in ["data", "results", "items", "records"]:
                        if key in data and isinstance(data[key], list):
                            self.current_df = pd.DataFrame(data[key])
                            break
                    else:
                        self.current_df = pd.json_normalize(data)
            else:
                self.current_df = pd.read_csv(StringIO(response.text))

            self.history.append(f"Extracted from API: {url}")
            return f"Loaded {len(self.current_df)} rows from API\nColumns: {list(self.current_df.columns)}"
        except Exception as e:
            return f"API extraction error: {str(e)}"

    def filter_data(self, column: str, keyword: str) -> str:
        if self.current_df is None:
            return "No data loaded. Extract data first."
        try:
            if column not in self.current_df.columns:
                return f"Column '{column}' not found. Available: {list(self.current_df.columns)}"
            original_count = len(self.current_df)
            self.current_df[column] = self.current_df[column].astype(str).str.lower()
            matches = self.current_df[
                self.current_df[column].str.contains(keyword.lower(), na=False)
            ]
            if len(matches) == 0:
                sample_values = self.current_df[column].unique()[:5]
                return f"No matches for '{keyword}' in '{column}'\nSample values: {sample_values}"
            self.current_df = matches
            self.history.append(f"Filtered by {column}='{keyword}'")
            return f"Filtered: {len(self.current_df)}/{original_count} rows match '{keyword}' in '{column}'"
        except Exception as e:
            return f"Filter error: {str(e)}"

    def load_to_postgres(self, table_name: str, db_url: str = None) -> str:
        if self.current_df is None:
            return "No data to load. Please extract data first."
        try:
            db_url = db_url or self.default_db_url
            if not db_url:
                return "No database URL provided and no default set"
            df_to_load = self.current_df.copy()

            dtype_mapping = {}
            for col in df_to_load.columns:
                if pd.api.types.is_object_dtype(df_to_load[col]):
                    if (
                        df_to_load[col]
                        .apply(lambda x: isinstance(x, (dict, list)))
                        .any()
                    ):
                        df_to_load[col] = df_to_load[col].apply(
                            lambda x: json.dumps(x)
                            if isinstance(x, (dict, list))
                            else x
                        )
                        dtype_mapping[col] = JSON()
                    else:
                        dtype_mapping[col] = Text()
                elif pd.api.types.is_integer_dtype(df_to_load[col]):
                    dtype_mapping[col] = BigInteger()
                elif pd.api.types.is_float_dtype(df_to_load[col]):
                    dtype_mapping[col] = Float()
                elif pd.api.types.is_bool_dtype(df_to_load[col]):
                    dtype_mapping[col] = Boolean()
                elif pd.api.types.is_datetime64_any_dtype(df_to_load[col]):
                    dtype_mapping[col] = DateTime()

            engine = create_engine(db_url)
            df_to_load.to_sql(
                table_name,
                engine,
                if_exists="replace",
                index=False,
                dtype=dtype_mapping,
            )

            self.history.append(f"Loaded to {table_name}")
            return f"Successfully loaded {len(df_to_load)} rows to '{table_name}'"
        except Exception as e:
            return f"Database error: {str(e)}"

    def get_data_info(self) -> str:
        if self.current_df is None:
            return "No data currently loaded"
        return f"Current dataset: {len(self.current_df)} rows, {len(self.current_df.columns)} columns\nColumns: {list(self.current_df.columns)}\nHistory: {' → '.join(self.history)}"

    def preview_data(self, rows: int = 5) -> str:
        if self.current_df is None:
            return "No data to preview"
        return f"Preview (first {rows} rows):\n{self.current_df.head(rows).to_string(index=False)}"
