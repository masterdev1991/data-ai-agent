import streamlit as st
import pandas as pd
import os

from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from elt_platform.llm.ollama_client import (
    AgentState,
    create_agent_graph,
)

# Page config
st.set_page_config(page_title="ETL Assistant", layout="wide")

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploaded_files")
UPLOADS_DIR.mkdir(exist_ok=True)

# Init session state
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState(
        messages=[
            SystemMessage(
                content=(
                    "You are an AI ETL assistant with dbt integration capabilities. You have tools for:\n"
                    "- Extracting CSV/API data\n"
                    "- Filtering and previewing data\n"
                    "- Loading to PostgreSQL\n"
                    "- Generating dbt sources.yml files\n"
                    "Use ONE tool per request to keep responses clear."
                )
            )
        ]
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_file_info" not in st.session_state:
    st.session_state.uploaded_file_info = None


@st.cache_resource
def get_agent_graph():
    return create_agent_graph()


def save_uploaded_file(uploaded_file):
    """
    Save uploaded file and return the file path
    """
    try:
        # Create a safe filename
        safe_filename = uploaded_file.name.replace(" ", "_")
        file_path = UPLOADS_DIR / safe_filename

        # Write the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Verify file was written and can be read
        if file_path.exists() and file_path.stat().st_size > 0:
            # Test if it's a valid CSV
            try:
                # test_df = pd.read_csv(file_path, nrows=1)
                return str(file_path.absolute())
            except Exception as e:
                st.error(f"Invalid CSV file: {e}")
                return None
        else:
            st.error("Failed to save uploaded file")
            return None

    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def process_user_message(user_input: str, agent_state: AgentState):
    """
    Process user message through the agent
    """
    app = get_agent_graph()

    # Clear previous conversation except system message
    agent_state.messages = [agent_state.messages[0], HumanMessage(content=user_input)]

    try:
        final_state = None
        progress_bar = st.progress(0, "Processing...")

        # Debug: Show what message is being sent to agent
        with st.expander("Debug: Message sent to agent", expanded=False):
            st.code(f"User Input: {user_input}")

        step = 0
        for output in app.stream(agent_state):
            progress_bar.progress(min(step * 0.3, 0.9), "Processing...")
            step += 1

            # Debug: Show agent processing steps
            with st.expander("Debug: Agent processing steps", expanded=False):
                st.write(f"Step {step}:", output)

            for _, value in output.items():
                if isinstance(value, AgentState):
                    final_state = value
                else:
                    final_state = AgentState(**value)

        progress_bar.progress(1.0, "Complete!")
        progress_bar.empty()

        # Get final LLM response
        if final_state and final_state.messages:
            for msg in reversed(final_state.messages):
                if (
                    isinstance(msg, AIMessage)
                    and msg.content
                    and not getattr(msg, "tool_calls", None)
                ):
                    response = msg.content
                    break
            else:
                response = "Task completed successfully!"
        else:
            response = "No response generated."

        # Update session state
        if final_state:
            st.session_state.agent_state.current_df = final_state.current_df
            st.session_state.agent_state.history = final_state.history
            st.session_state.agent_state.last_loaded_table = (
                final_state.last_loaded_table
            )
            st.session_state.agent_state.last_loaded_schema = (
                final_state.last_loaded_schema
            )

        return response

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return f"Error: {str(e)}"


def display_dataframe_info(df):
    """
    Display comprehensive dataframe information
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric(
            "Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )
    with col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")


# ========== UI ==========
st.title("ETL Assistant")
st.markdown(
    "Upload CSV files, process data, and generate dbt sources with AI assistance!"
)

# Sidebar
with st.sidebar:
    st.header("📁 File Upload")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file to extract and process data",
    )

    if uploaded_file is not None:
        # Save the uploaded file
        file_path = save_uploaded_file(uploaded_file)

        if file_path:
            st.session_state.uploaded_file_info = {
                "name": uploaded_file.name,
                "path": file_path,
                "size": uploaded_file.size,
            }

            st.success(f"File uploaded: {uploaded_file.name}")
            st.info(f"Saved to: {file_path}")
            st.info(f"Size: {uploaded_file.size / 1024:.1f} KB")

            # Quick actions
            st.header("⚡ Quick Actions")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "📊 Extract Data", use_container_width=True, type="primary"
                ):
                    with st.spinner("Extracting data..."):
                        # Use the exact file path from session state
                        exact_path = st.session_state.uploaded_file_info["path"]

                        # Try different command formats to see which one works
                        st.write("🔍 **Attempting extraction with exact path:**")
                        st.code(exact_path)

                        # Use the exact Windows path as-is first
                        response = process_user_message(
                            f'extract_csv_tool("{exact_path}")',
                            st.session_state.agent_state,
                        )

                        st.session_state.chat_history.extend(
                            [
                                ("User", f"Extract CSV data from {uploaded_file.name}"),
                                ("Assistant", response),
                            ]
                        )
                        st.rerun()

            with col2:
                if st.button("Preview", use_container_width=True):
                    with st.spinner("Loading preview..."):
                        response = process_user_message(
                            "preview data", st.session_state.agent_state
                        )
                        st.session_state.chat_history.extend(
                            [("User", "Preview data"), ("Assistant", response)]
                        )
                        st.rerun()

    # Current data info
    st.header("📋 Current Dataset")
    if st.session_state.agent_state.current_df is not None:
        df = st.session_state.agent_state.current_df

        st.metric("Rows", f"{len(df):,}")
        st.metric("Columns", len(df.columns))

        if st.button("Detailed Info", use_container_width=True):
            response = process_user_message(
                "get data info", st.session_state.agent_state
            )
            st.session_state.chat_history.extend(
                [("User", "Get detailed data info"), ("Assistant", response)]
            )
            st.rerun()

        # Additional quick actions for loaded data
        st.subheader("Data Operations")

        if st.button("Filter Data", use_container_width=True):
            st.session_state.show_filter_form = True

        # Table name input for loading to postgres
        table_name = st.text_input("Table name for PostgreSQL", placeholder="my_table")
        if st.button(
            "🗄️ Load to PostgreSQL", use_container_width=True, disabled=not table_name
        ):
            if table_name:
                response = process_user_message(
                    f"load to postgres {table_name}", st.session_state.agent_state
                )
                st.session_state.chat_history.extend(
                    [
                        ("User", f"Load data to PostgreSQL table '{table_name}'"),
                        ("Assistant", response),
                    ]
                )
                st.rerun()
    else:
        st.info("No data loaded yet. Upload and extract a CSV file to get started.")

    # Direct CSV extraction test
    if st.session_state.uploaded_file_info:
        st.header("🧪 Direct CSV Test")
        if st.button("Test Direct CSV Read"):
            file_path = st.session_state.uploaded_file_info["path"]
            try:
                # Test reading the CSV directly with pandas
                test_df = pd.read_csv(file_path)
                st.success("Direct CSV read successful!")
                st.write(f"Shape: {test_df.shape}")
                st.dataframe(test_df.head(3))

                # Now try to manually set it to agent state for testing
                st.session_state.agent_state.current_df = test_df
                st.session_state.agent_state.history.append(
                    f"Direct loaded: {st.session_state.uploaded_file_info['name']}"
                )
                st.success("Manually set dataframe in agent state")

            except Exception as e:
                st.error(f"Direct CSV read failed: {e}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Chat interface
    st.header("Chat with ETL Assistant")

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message("user" if role == "User" else "assistant"):
                st.write(message)

    # Chat input
    if user_input := st.chat_input(
        "Ask me to extract, filter, load, or generate dbt sources..."
    ):
        # Process the user input to handle file path references
        processed_input = user_input

        # If user mentions "uploaded file" and we have a file, use the actual path
        # response = process_user_message(
        #     f'extract_csv_tool("{exact_path}")',
        #     st.session_state.agent_state,
        # )
        if (
            "uploaded file" in user_input.lower() or "upload" in user_input.lower()
        ) and st.session_state.uploaded_file_info:
            file_path = st.session_state.uploaded_file_info["path"].replace("\\", "/")
            processed_input = f'extract_csv_tool("{file_path}")'

        # Add user message
        st.session_state.chat_history.append(("User", user_input))

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            response = process_user_message(
                processed_input, st.session_state.agent_state
            )
            st.write(response)

        # Add assistant response
        st.session_state.chat_history.append(("Assistant", response))
        st.rerun()

with col2:
    # Command suggestions
    with st.expander("💡 Example Commands", expanded=False):
        st.markdown("""
        **File Operations:**
        - `extract csv from {st.session_state.uploaded_file_info["path"].replace("\\", "/") if st.session_state.uploaded_file_info else "your_file.csv"}`
        - `extract data from uploaded file`
        - `preview data with 10 rows`
        - `get data info`
        
        **Data Processing:**
        - `filter data where status equals 'active'`
        - `filter data where price contains '100'`
        
        **Database Operations:**
        - `load data to postgres table customers`
        - `generate dbt sources for customers`
        
        **API Operations:**
        - `extract data from https://api.example.com/data`
        """)

# Data preview section
if st.session_state.agent_state.current_df is not None:
    st.header("Current Dataset Overview")

    df = st.session_state.agent_state.current_df

    # Display metrics
    display_dataframe_info(df)

    # Preview controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        preview_rows = st.number_input(
            "Rows to preview", min_value=1, max_value=100, value=10
        )

    with col2:
        show_dtypes = st.checkbox("Show data types", value=False)

    with col3:
        if st.button("Refresh Preview", use_container_width=True):
            st.rerun()

    # Data preview
    st.subheader("Data Preview")
    preview_df = df.head(preview_rows)

    if show_dtypes:
        # Create a styled dataframe with type information
        styled_df = preview_df.copy()
        type_row = pd.Series(
            {col: f"({df[col].dtype})" for col in df.columns}, name="Data Types"
        )
        display_df = pd.concat(
            [pd.DataFrame([type_row]), styled_df], ignore_index=False
        )
        st.dataframe(display_df, use_container_width=True)
    else:
        st.dataframe(preview_df, use_container_width=True)

    # Column information
    if st.checkbox("Show Column Details"):
        col_info = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes,
                "Non-Null Count": df.count(),
                "Null Count": df.isnull().sum(),
                "Unique Values": df.nunique(),
            }
        )
        st.dataframe(col_info, use_container_width=True)

# Clear chat history button
if st.session_state.chat_history:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**ETL Assistant**")

# Debug information (only show in development)
if st.checkbox("Show Debug Info"):
    st.subheader("Debug Information")
    st.write("**Uploaded File Info:**", st.session_state.uploaded_file_info)
    if st.session_state.uploaded_file_info:
        st.code(f"Exact path to use: {st.session_state.uploaded_file_info['path']}")
        st.code(
            f"Normalized path: {st.session_state.uploaded_file_info['path'].replace('\\', '/')}"
        )
        # Test if file exists

        file_exists = os.path.exists(st.session_state.uploaded_file_info["path"])
        st.write(f"**File exists:** {file_exists}")
        if file_exists:
            st.write(
                f"**File size on disk:** {os.path.getsize(st.session_state.uploaded_file_info['path'])} bytes"
            )
    st.write("**Agent State History:**", st.session_state.agent_state.history)
    st.write(
        "**Current DF Shape:**",
        st.session_state.agent_state.current_df.shape
        if st.session_state.agent_state.current_df is not None
        else "None",
    )
