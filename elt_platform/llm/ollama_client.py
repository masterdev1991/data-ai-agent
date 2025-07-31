from datetime import datetime
import json
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from elt_platform.agents.dbt_sources_generator_agent import DBTSourcesGenerator
from elt_platform.agents.ingestion_agent import IngestionAgent
from elt_platform.core.settings import get_settings

settings = get_settings()


# ========== AGENT STATE ==========
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    current_df: Optional[pd.DataFrame] = None
    history: List[str] = Field(default_factory=list)
    default_db_url: Optional[str] = settings.DATABASE_URL
    last_loaded_table: Optional[str] = None
    last_loaded_schema: Optional[str] = "public"

    class Config:
        arbitrary_types_allowed = True


# ========== INGESTOR ==========
ingestor = IngestionAgent()
dbt_generator = DBTSourcesGenerator(settings.DATABASE_URL)


# ========== TOOLS ==========
@tool
def extract_csv_tool(filename: str) -> str:
    """
    Extract data from a local CSV file or a remote CSV URL.

    Args:
        filename (str): Path to a local CSV file or a URL to a CSV file.

    Returns:
        str: A message indicating success or failure of the extraction.
    """
    return ingestor.extract_csv(filename)


@tool
def extract_api_tool(url: str, headers: str = None) -> str:
    """
    Extract JSON data from an external API endpoint.

    Args:
        url (str): The API endpoint URL.
        headers (str, optional): JSON string of request headers, if required.

    Returns:
        str: A message indicating success or failure of the API data extraction.
    """
    return ingestor.extract_api(url, headers)


@tool
def filter_data_tool(column: str, keyword: str) -> str:
    """
    Filter the current dataset to rows where the given column contains the specified keyword.

    Args:
        column (str): Column name to apply the filter on.
        keyword (str): Keyword to filter for within the column.

    Returns:
        str: A summary or preview of the filtered data.
    """
    return ingestor.filter_data(column, keyword)


@tool
def load_to_postgres_tool(table_name: str, db_url: str = None) -> str:
    """
    Load the current dataset into a PostgreSQL table.

    Args:
        table_name (str): Name of the target table in the database.
        db_url (str, optional): SQLAlchemy-compatible DB URL (defaults to internal config).

    Returns:
        str: A message indicating success or failure of the load operation.
    """
    return ingestor.load_to_postgres(table_name, db_url)


@tool
def preview_data_tool(rows: int = 5) -> str:
    """
    Show a preview of the current dataset.

    Args:
        rows (int, optional): Number of rows to display. Defaults to 5.

    Returns:
        str: A string preview of the dataset.
    """
    return ingestor.preview_data(rows)


@tool
def get_data_info_tool() -> str:
    """
    Display general information about the current dataset, including shape and column types.

    Returns:
        str: A summary of the dataset's structure and content.
    """
    return ingestor.get_data_info()


@tool
def generate_dbt_sources_tool(
    table_name: str, schema_name: str = "public", source_file_path: str = None
) -> str:
    """
    Generate a dbt sources.yml file for the specified table that was loaded into PostgreSQL.

    Args:
        table_name (str): Name of the table in the database to generate sources for.
        schema_name (str, optional): Database schema name. Defaults to "public".
        source_file_path (str, optional): Original source file path for documentation.

    Returns:
        str: A message indicating success or failure of the dbt sources generation.
    """
    try:
        # Generate the sources YAML
        sources_yaml = dbt_generator.generate_sources_yaml(
            table_name=table_name,
            schema_name=schema_name,
            source_file_path=source_file_path,
        )

        # Save the file
        file_path = dbt_generator.save_sources_file(sources_yaml, table_name)

        # Get table info for summary
        table_schema = dbt_generator.introspect_table_schema(schema_name, table_name)

        return (
            f"Successfully generated dbt sources.yml for table '{table_name}'\n"
            f"File saved to: {file_path}\n"
            f"Table has {len(table_schema['columns'])} columns and {table_schema['row_count']} rows\n"
            f"Schema: {schema_name}\n"
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    except Exception as e:
        return f"Error generating dbt sources for table '{table_name}': {str(e)}"


tools = [
    extract_csv_tool,
    extract_api_tool,
    filter_data_tool,
    load_to_postgres_tool,
    preview_data_tool,
    get_data_info_tool,
    generate_dbt_sources_tool,
]


# ========== LLM ==========
def create_llm():
    llm = ChatOllama(model="smollm2:latest", temperature=0.1).bind_tools(tools)
    # print("🛠 Registered tools:", [t.name for t in tools])
    return llm


# ========== AGENT NODE ==========
def agent_node(state: AgentState) -> Dict[str, Any]:
    llm = create_llm()
    llm_no_tools = ChatOllama(model="smollm2:latest", temperature=0.1)

    if not state.messages:
        return {"messages": state.messages}

    last_msg = state.messages[-1]
    if isinstance(last_msg, (HumanMessage, ToolMessage)):
        response = llm.invoke(state.messages)

        # Handle unhelpful fallback response
        if (
            isinstance(response, AIMessage)
            and not getattr(response, "tool_calls", None)
            and "cannot be answered with the provided tools" in (response.content or "")
        ):
            # Try again with normal LLM
            response = llm_no_tools.invoke(state.messages)
        return {"messages": state.messages + [response]}

    return {"messages": state.messages}


# ========== TOOL NODE ==========
def tool_node(state: AgentState) -> Dict[str, Any]:
    if not state.messages:
        return {"messages": state.messages}

    last_message = state.messages[-1]

    tool_calls = getattr(last_message, "tool_calls", None) or (
        getattr(last_message, "additional_kwargs", {}).get("tool_calls")
    )

    # print(f"Raw tool_calls: {tool_calls}")

    if not tool_calls:
        return {"messages": state.messages}

    # ONLY PROCESS THE FIRST TOOL CALL
    tool_call = tool_calls[0]  # Take only the first tool call

    try:
        if isinstance(tool_call, dict):
            func_name = tool_call.get("name") or tool_call.get("function", {}).get(
                "name"
            )
            args = tool_call.get("args") or json.loads(
                tool_call.get("function", {}).get("arguments", "{}")
            )
            tool_call_id = tool_call.get("id", "")
        else:
            func_name = getattr(tool_call, "name", "")
            args = json.loads(getattr(tool_call, "arguments", "{}"))
            tool_call_id = getattr(tool_call, "id", "")

        if not func_name:
            raise ValueError("Tool call missing function name.")

        print(f"Executing tool: {func_name} with args: {args}")

        tool_func = next((t for t in tools if t.name == func_name), None)
        if tool_func is None:
            raise ValueError(f"No tool found with name: {func_name}")

        result = tool_func.invoke(args)

        # Update state based on tool execution
        if func_name == "extract_csv_tool" and "Loaded" in result:
            try:
                df = (
                    ingestor._extract_from_url(args["filename"])
                    if args["filename"].startswith(("http", "https"))
                    else ingestor._extract_from_file(args["filename"])
                )
                state.current_df = df
                state.history.append(f"Extracted {args['filename']}")
            except Exception as e:
                result = f"Error processing file: {str(e)}"

        elif func_name == "filter_data_tool" and "Filtered:" in result:
            state.history.append(f"Filtered by {args['column']}='{args['keyword']}'")

        elif (
            func_name == "load_to_postgres_tool"
            and "successfully loaded" in result.lower()
        ):
            # Store the loaded table info for potential dbt source generation
            state.last_loaded_table = args["table_name"]
            state.last_loaded_schema = "public"  # Default schema
            state.history.append(f"Loaded to PostgreSQL table: {args['table_name']}")

        tool_message = ToolMessage(content=result, tool_call_id=tool_call_id)

    except Exception as e:
        print(f"Error in tool execution: {str(e)}")
        tool_message = ToolMessage(
            content=f"Error processing tool call: {str(e)}",
            tool_call_id=tool_call_id if "tool_call_id" in locals() else "",
        )

    return {"messages": state.messages + [tool_message]}


# ========== CONTROL FLOW ==========
def should_continue(state: AgentState) -> str:
    if not state.messages:
        return "end"

    last = state.messages[-1]

    # If last message is from agent and has tool calls, continue to tools
    if isinstance(last, AIMessage):
        has_tool_calls = getattr(last, "tool_calls", None) or getattr(
            last, "additional_kwargs", {}
        ).get("tool_calls")
        print("[should_continue] Tool Calls Present:", bool(has_tool_calls))
        return "tools" if has_tool_calls else "end"

    # If last message is a tool result, go back to agent for final response
    elif isinstance(last, ToolMessage):
        print("[should_continue] Tool completed, generating final response")
        return "agent"

    return "end"


# ========== FINAL RESPONSE NODE ==========
def final_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate a final response after tool execution
    """
    llm = ChatOllama(model="smollm2:latest", temperature=0.1)  # No tools bound

    # Add a system message to generate a helpful response
    messages = state.messages + [
        HumanMessage(
            content="Please provide a helpful summary of what was accomplished. Using the Pronouns 'I', Because you are the agent"
        )
    ]

    response = llm.invoke(messages)
    return {"messages": state.messages + [response]}


# ========== GRAPH ==========
def create_agent_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("final_response", final_response_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Define edges
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )

    graph.add_conditional_edges(
        "tools", should_continue, {"agent": "final_response", "end": END}
    )
    # graph.add_conditional_edges(
    #     "agent", should_continue, {"tools": "tools", "final_response": "final_response"}
    # )

    # graph.add_conditional_edges(
    #     "tools", should_continue, {"agent": "final_response", "end": END}
    # )

    graph.add_edge("final_response", END)

    return graph.compile()


# ========== MAIN ==========
def run_chat():
    app = create_agent_graph()
    state = AgentState(
        messages=[
            SystemMessage(
                content=(
                    "You are an AI ETL assistant with dbt integration capabilities. You have access to tools for:\n"
                    "- Extracting data from CSV files and APIs\n"
                    "- Filtering and previewing data\n"
                    "- Loading data to PostgreSQL\n"
                    "- Generating dbt sources.yml files for loaded tables\n\n"
                    "When a user loads data to PostgreSQL, you can also generate dbt sources configuration files "
                    "to help them set up their dbt project. Use only ONE tool per request to keep responses focused and clear."
                    "When a user asks you to perform a task, use the appropriate tool and provide a clear response. "
                    "Use only ONE tool per request to keep responses focused and clear."
                )
            )
        ]
    )

    print("ETL Assistant initialized. Type 'exit' to quit.")
    print("💡 Available commands:")
    print("   - Extract CSV: 'extract <filename>'")
    print("   - Load to database: 'load to postgres <table_name>'")
    print("   - Generate dbt sources: 'generate dbt sources for <table_name>'")
    print("   - Preview data: 'preview data'")
    print("   - And more...")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Clear previous conversation except system message
        state.messages = [state.messages[0], HumanMessage(content=user_input)]

        # state.messages.append(HumanMessage(content=user_input))

        try:
            final_state = None
            for output in app.stream(state):
                for key, value in output.items():
                    if isinstance(value, AgentState):
                        final_state = value
                    else:
                        final_state = AgentState(**value)

            # Print the final AI response
            if final_state and final_state.messages:
                for msg in reversed(final_state.messages):
                    if (
                        isinstance(msg, AIMessage)
                        and msg.content
                        and not getattr(msg, "tool_calls", None)
                    ):
                        print(f"\nAssistant: {msg.content}")
                        break

            # Update state for next iteration (preserve current_df and history)
            if final_state:
                state.current_df = final_state.current_df
                state.history = final_state.history

        except Exception as e:
            print(f"\nError: {str(e)}")
            continue


if __name__ == "__main__":
    run_chat()
