import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, Union
import os
from dotenv import load_dotenv
import sqlite3
import duckdb
from pathlib import Path
import tempfile
import json
from datetime import datetime
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect
import numpy as np

# Import our chatbot components
from chatbot import setup_chatbot, DataFrameManager


class DatabaseManager:
    # Unchanged; use your existing implementation for this class
    ...


class StreamlitChatBot:
    def __init__(self):
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        # Initialize session state
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = setup_chatbot(os.getenv("OPENAI_API_KEY"))
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'current_table' not in st.session_state:
            st.session_state.current_table = None

    def setup_page(self):
        """Set up the Streamlit page layout."""
        st.set_page_config(
            page_title="AI ChatBot with Data Management",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("AI-Powered ChatBot with Data Management")

    def render_sidebar(self):
        """Render the sidebar with enhanced data upload and table selection options."""
        with st.sidebar:
            st.header("Data Management")
            
            uploaded_file = st.file_uploader(
                "Upload Data File",
                type=['csv', 'xlsx', 'xls', 'db'],
                help="Upload a CSV, Excel, or SQLite database file"
            )
            
            if uploaded_file:
                df = self.read_file(uploaded_file)
                
                if df is not None:
                    table_name = st.text_input(
                        "Table Name",
                        value=uploaded_file.name.split('.')[0]
                    )
                    description = st.text_area(
                        "Description",
                        value=f"Data from {uploaded_file.name}"
                    )
                    
                    if st.button("Add Table"):
                        if self.db_manager.save_dataframe(
                            df,
                            table_name,
                            description,
                            uploaded_file.name.split('.')[-1]
                        ):
                            st.session_state.chatbot.df_manager.add_dataframe(
                                name=table_name,
                                df=df,
                                description=description
                            )
                            st.success(f"Successfully added table: {table_name}")
                            st.session_state.current_table = table_name

            tables_df = self.db_manager.get_all_tables()
            if not tables_df.empty:
                st.subheader("Available Tables")
                for _, row in tables_df.iterrows():
                    with st.expander(f"📊 {row['table_name']}"):
                        st.write(f"Description: {row['description']}")
                        st.write(f"Rows: {row['row_count']}")
                        st.write(f"Columns: {row['column_count']}")
                        st.write(f"Upload Date: {row['upload_date']}")
                        
                        if st.button(f"View Details", key=f"view_{row['table_name']}"):
                            metadata = self.db_manager.get_table_metadata(row['table_name'])
                            st.json(metadata)

    def render_chat_interface(self):
        """Render the main chat interface."""
        st.header("Chat Interface")
        user_input = st.text_input("Ask a question:")
        
        if user_input:
            response = st.session_state.chatbot.process_question(user_input)
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.write("### Query:")
                st.code(response["query"])
                st.write("### Results:")
                st.dataframe(pd.DataFrame(response["data"]))
                st.write("### Summary:")
                st.info(response["summary"])
                
                st.session_state.chat_history.append({
                    "question": user_input,
                    "response": response
                })
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, entry in enumerate(st.session_state.chat_history):
                st.write(f"**Q{i+1}: {entry['question']}**")
                st.write(f"**A{i+1}: {entry['response']['summary']}**")

    def read_file(self, uploaded_file) -> Union[pd.DataFrame, None]:
        """Read different file types into a DataFrame."""
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            if file_type == 'csv':
                return pd.read_csv(uploaded_file)
            elif file_type in ['xls', 'xlsx']:
                return pd.read_excel(uploaded_file)
            elif file_type == 'db':
                with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                temp_engine = create_engine(f'sqlite:///{tmp_path}')
                inspector = inspect(temp_engine)
                table_names = inspector.get_table_names()
                if not table_names:
                    st.error("No tables found in the database file")
                    return None
                selected_table = st.selectbox("Select a table to import:", table_names)
                if selected_table:
                    df = pd.read_sql_table(selected_table, temp_engine)
                    os.unlink(tmp_path)
                    return df
            else:
                st.error(f"Unsupported file type: {file_type}")
                return None
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitChatBot()
    app.setup_page()
    app.render_sidebar()
    app.render_chat_interface()


if __name__ == "__main__":
    main()
