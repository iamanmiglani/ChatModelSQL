import os
import pandas as pd
import sqlite3
import duckdb
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import openai
from datetime import datetime
import json
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please ensure it is set in the .env file.")

@dataclass
class DataFrameMetadata:
    """Stores metadata about each DataFrame for better context handling"""
    name: str
    columns: List[str]
    sample_values: Dict[str, List[Any]]
    cardinality: Dict[str, int]
    total_rows: int
    description: str = ""
    source_type: str = ""  # Added to track data source type
    last_updated: datetime = datetime.now()

class DataFrameManager:
    """Manages multiple DataFrames using both DuckDB and SQLite"""

    def __init__(self, sql_db_path: str = "app_data.db"):
        self.duckdb_conn = duckdb.connect(database=':memory:', read_only=False)
        self.sql_engine = create_engine(f'sqlite:///{sql_db_path}')
        self.metadata: Dict[str, DataFrameMetadata] = {}
        self.setup_logging()
        self.load_existing_tables()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_existing_tables(self):
        """Load existing tables from SQLite database into DuckDB"""
        try:
            # Get list of tables from SQLite
            with self.sql_engine.connect() as conn:
                tables = pd.read_sql(
                    text("SELECT name FROM sqlite_master WHERE type='table'"),
                    conn
                )

                # Skip metadata tables
                tables = tables[~tables['name'].isin(['table_metadata', 'column_metadata'])]

                for table_name in tables['name']:
                    # Load table from SQLite
                    df = pd.read_sql_table(table_name, conn)

                    # Get metadata from metadata tables
                    metadata = pd.read_sql(
                        text("SELECT * FROM table_metadata WHERE table_name = :table_name"),
                        conn,
                        params={"table_name": table_name}
                    ).iloc[0]

                    column_metadata = pd.read_sql(
                        text("SELECT * FROM column_metadata WHERE table_name = :table_name"),
                        conn,
                        params={"table_name": table_name}
                    )

                    # Register with DuckDB
                    self.duckdb_conn.register(table_name, df)

                    # Reconstruct metadata
                    self.metadata[table_name] = DataFrameMetadata(
                        name=table_name,
                        columns=df.columns.tolist(),
                        sample_values={row['column_name']: json.loads(row['sample_values'])
                                       for _, row in column_metadata.iterrows()},
                        cardinality={row['column_name']: row['cardinality']
                                     for _, row in column_metadata.iterrows()},
                        total_rows=len(df),
                        description=metadata['description'],
                        source_type=metadata['file_type'],
                        last_updated=metadata['upload_date']
                    )

                    self.logger.info(f"Loaded existing table '{table_name}' from SQLite")

        except Exception as e:
            self.logger.error(f"Error loading existing tables: {str(e)}")

    def add_dataframe(self, name: str, df: pd.DataFrame, description: str = "", source_type: str = "") -> None:
        """Add a DataFrame to both DuckDB and SQLite with metadata"""
        try:
            # Clean table name
            name = "".join(c if c.isalnum() else "_" for c in name)

            # Save to SQLite
            df.to_sql(name, self.sql_engine, if_exists='replace', index=False)

            # Register with DuckDB
            self.duckdb_conn.register(name, df)

            # Calculate metadata
            cardinality = {col: df[col].nunique() for col in df.columns}
            sample_values = {
                col: df[col].dropna().sample(min(5, len(df))).tolist()
                for col in df.columns
            }

            # Store metadata
            metadata = DataFrameMetadata(
                name=name,
                columns=df.columns.tolist(),
                sample_values=sample_values,
                cardinality=cardinality,
                total_rows=len(df),
                description=description,
                source_type=source_type,
                last_updated=datetime.now()
            )
            self.metadata[name] = metadata

            # Save metadata to SQLite
            with self.sql_engine.connect() as conn:
                # Table metadata
                conn.execute(text("""
                    INSERT OR REPLACE INTO table_metadata 
                    (table_name, description, upload_date, file_type, row_count, column_count)
                    VALUES (:table_name, :description, :upload_date, :file_type, :row_count, :column_count)
                """), {
                    "table_name": name,
                    "description": description,
                    "upload_date": metadata.last_updated,
                    "file_type": source_type,
                    "row_count": metadata.total_rows,
                    "column_count": len(metadata.columns)
                })

                # Column metadata
                for col in df.columns:
                    conn.execute(text("""
                        INSERT OR REPLACE INTO column_metadata 
                        (table_name, column_name, data_type, cardinality, sample_values)
                        VALUES (:table_name, :column_name, :data_type, :cardinality, :sample_values)
                    """), {
                        "table_name": name,
                        "column_name": col,
                        "data_type": str(df[col].dtype),
                        "cardinality": cardinality[col],
                        "sample_values": json.dumps(sample_values[col])
                    })

                conn.commit()

            self.logger.info(f"Successfully added DataFrame '{name}' with {len(df)} rows")

        except Exception as e:
            self.logger.error(f"Error adding DataFrame '{name}': {str(e)}")
            raise

class QueryGenerator:
    """Generates SQL queries based on natural language input"""

    def __init__(self, df_manager: DataFrameManager):
        self.df_manager = df_manager
        self.openai_client = openai.Client(api_key=OPENAI_API_KEY)

    def generate_query(self, user_question: str) -> Tuple[str, str]:
        """Generate SQL query and summary prompt from user question"""
        context = self._create_context()

        messages = [
            {"role": "system", "content": f"""You are a SQL expert. Generate a SQL query based on the following context and question. 
             The tables are stored in a DuckDB database. Only return the SQL query, nothing else.
             Available tables and their metadata:
             {context}"""},
            {"role": "user", "content": user_question}
        ]

        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0
        )

        sql_query = response.choices[0].message.content
        summary_prompt = f"Summarize the following data in a concise, business-friendly way: [QUERY_RESULT]"

        return sql_query, summary_prompt

    def _create_context(self) -> str:
        """Create enhanced context string from DataFrame metadata"""
        context = []
        for name, meta in self.df_manager.metadata.items():
            table_info = [
                f"Table: {name}",
                f"Description: {meta.description}",
                f"Source Type: {meta.source_type}",
                f"Last Updated: {meta.last_updated}",
                f"Total Rows: {meta.total_rows}",
                "Columns:"
            ]

            for col in meta.columns:
                samples = meta.sample_values[col]
                cardinality = meta.cardinality[col]
                table_info.append(f"  - {col} (unique values: {cardinality}, samples: {samples})")

            context.append("\n".join(table_info))

        return "\n\n".join(context)

class ChatBot:
    """Main chatbot class that handles user interactions"""

    def __init__(self, df_manager: DataFrameManager, query_generator: QueryGenerator):
        self.df_manager = df_manager
        self.query_generator = query_generator

    def process_question(self, question: str) -> Dict[str, Any]:
        """Process user question and return results with summary"""
        try:
            sql_query, summary_prompt = self.query_generator.generate_query(question)

            # Execute query using DuckDB for performance
            result_df = self.df_manager.duckdb_conn.execute(sql_query).fetchdf()

            # Generate summary
            summary_messages = [
                {"role": "system", "content": "You are a data analyst. Provide a concise summary of the data."},
                {"role": "user", "content": summary_prompt.replace("[QUERY_RESULT]", result_df.to_string())}
            ]

            summary_response = self.query_generator.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=summary_messages,
                temperature=0.7
            )

            return {
                "query": sql_query,
                "data": result_df.to_dict(orient="records"),
                "summary": summary_response.choices[0].message.content,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Helper function to setup chatbot
def setup_chatbot(sql_db_path: str = "app_data.db") -> ChatBot:
    """Setup and initialize the chatbot with database connection"""
    df_manager = DataFrameManager(sql_db_path)
    query_generator = QueryGenerator(df_manager)
    return ChatBot(df_manager, query_generator)
