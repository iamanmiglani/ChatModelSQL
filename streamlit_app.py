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
    def __init__(self, db_path: str = "app_data.db"):
        """Initialize database connection and create necessary tables"""
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        self.setup_database()
        
    def setup_database(self):
        """Create necessary tables for storing metadata"""
        with self.engine.connect() as conn:
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS table_metadata (
                    table_name TEXT PRIMARY KEY,
                    description TEXT,
                    upload_date TIMESTAMP,
                    file_type TEXT,
                    row_count INTEGER,
                    column_count INTEGER
                )
            """))
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS column_metadata (
                    table_name TEXT,
                    column_name TEXT,
                    data_type TEXT,
                    cardinality INTEGER,
                    sample_values TEXT,
                    FOREIGN KEY (table_name) REFERENCES table_metadata(table_name)
                )
            """))
            conn.commit()

    def save_dataframe(self, df: pd.DataFrame, table_name: str, description: str, file_type: str) -> bool:
        """Save DataFrame to SQL database and store metadata"""
        try:
            # Clean table name to be SQL-friendly
            table_name = "".join(c if c.isalnum() else "_" for c in table_name)
            
            # Save DataFrame to SQL
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)
            
            # Store metadata
            metadata = {
                'table_name': table_name,
                'description': description,
                'upload_date': datetime.now(),
                'file_type': file_type,
                'row_count': len(df),
                'column_count': len(df.columns)
            }
            
            with self.engine.connect() as conn:
                # Delete existing metadata if any
                conn.execute(sa.text(
                    "DELETE FROM table_metadata WHERE table_name = :table_name"
                ), {"table_name": table_name})
                conn.execute(sa.text(
                    "DELETE FROM column_metadata WHERE table_name = :table_name"
                ), {"table_name": table_name})
                
                # Insert new metadata
                conn.execute(sa.text("""
                    INSERT INTO table_metadata 
                    (table_name, description, upload_date, file_type, row_count, column_count)
                    VALUES (:table_name, :description, :upload_date, :file_type, :row_count, :column_count)
                """), metadata)
                
                # Store column metadata
                for column in df.columns:
                    column_metadata = {
                        'table_name': table_name,
                        'column_name': column,
                        'data_type': str(df[column].dtype),
                        'cardinality': df[column].nunique(),
                        'sample_values': json.dumps(df[column].dropna().sample(min(5, len(df))).tolist())
                    }
                    conn.execute(sa.text("""
                        INSERT INTO column_metadata 
                        (table_name, column_name, data_type, cardinality, sample_values)
                        VALUES (:table_name, :column_name, :data_type, :cardinality, :sample_values)
                    """), column_metadata)
                
                conn.commit()
            
            return True
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
            return False

    def get_all_tables(self) -> pd.DataFrame:
        """Get metadata for all tables"""
        return pd.read_sql("SELECT * FROM table_metadata", self.engine)

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Get detailed metadata for a specific table"""
        table_info = pd.read_sql(
            "SELECT * FROM table_metadata WHERE table_name = :table_name",
            self.engine,
            params={"table_name": table_name}
        ).to_dict('records')[0]
        
        columns_info = pd.read_sql(
            "SELECT * FROM column_metadata WHERE table_name = :table_name",
            self.engine,
            params={"table_name": table_name}
        )
        
        return {
            **table_info,
            'columns': columns_info.to_dict('records')
        }

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

    def read_file(self, uploaded_file) -> Union[pd.DataFrame, None]:
        """Read different file types into a DataFrame"""
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == 'csv':
                return pd.read_csv(uploaded_file)
            elif file_type in ['xls', 'xlsx']:
                return pd.read_excel(uploaded_file)
            elif file_type == 'db':
                # Save uploaded SQLite database to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Connect to the temporary database
                temp_engine = create_engine(f'sqlite:///{tmp_path}')
                
                # Let user select a table
                inspector = inspect(temp_engine)
                table_names = inspector.get_table_names()
                
                if not table_names:
                    st.error("No tables found in the database file")
                    return None
                
                selected_table = st.selectbox("Select a table to import:", table_names)
                if selected_table:
                    df = pd.read_sql_table(selected_table, temp_engine)
                    os.unlink(tmp_path)  # Clean up temp file
                    return df
            else:
                st.error(f"Unsupported file type: {file_type}")
                return None
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None

    def render_sidebar(self):
        """Render the sidebar with enhanced data upload and table selection options"""
        with st.sidebar:
            st.header("Data Management")
            
            # File uploader with multiple file type support
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
                        # Save to SQL database
                        if self.db_manager.save_dataframe(
                            df,
                            table_name,
                            description,
                            uploaded_file.name.split('.')[-1]
                        ):
                            # Add to chatbot's DataFrame manager
                            st.session_state.chatbot.df_manager.add_dataframe(
                                name=table_name,
                                df=df,
                                description=description
                            )
                            st.success(f"Successfully added table: {table_name}")
                            st.session_state.current_table = table_name
            
            # Display available tables
            st.subheader("Available Tables")
            tables_df = self.db_manager.get_all_tables()
            
            if not tables_df.empty:
                for _, row in tables_df.iterrows():
                    with st.expander(f"📊 {row['table_name']}"):
                        st.write(f"Description: {row['description']}")
                        st.write(f"Rows: {row['row_count']}")
                        st.write(f"Columns: {row['column_count']}")
                        st.write(f"Upload Date: {row['upload_date']}")
                        
                        if st.button(f"View Details", key=f"view_{row['table_name']}"):
                            metadata = self.db_manager.get_table_metadata(row['table_name'])
                            st.json(metadata)

    # ... (rest of the StreamlitChatBot class remains the same)

def main():
    # Initialize the Streamlit chatbot
    app = StreamlitChatBot()
    
    # Setup the page
    app.setup_page()
    
    # Create two columns: sidebar and main content
    app.render_sidebar()
    
    # Render the main chat interface
    app.render_chat_interface()

if __name__ == "__main__":
    main()
