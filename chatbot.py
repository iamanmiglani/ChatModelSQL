import os
import pandas as pd
import duckdb
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import openai
from datetime import datetime
import json
import logging
from sqlalchemy import create_engine, text


@dataclass
class DataFrameMetadata:
    """Stores metadata about each DataFrame for better context handling."""
    name: str
    columns: List[str]
    sample_values: Dict[str, List[Any]]
    cardinality: Dict[str, int]
    total_rows: int
    description: str = ""
    source_type: str = ""
    last_updated: datetime = datetime.now()


class DataFrameManager:
    """Manages multiple DataFrames using both DuckDB and SQLite."""

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
        """Load existing tables from SQLite database into DuckDB."""
        try:
            with self.sql_engine.connect() as conn:
                tables = pd.read_sql(
                    text("SELECT name FROM sqlite_master WHERE type='table'"),
                    conn
                )

                if tables.empty:
                    self.logger.info("No tables found in the SQLite database.")
                    return

                for table_name in tables['name']:
                    df = pd.read_sql_table(table_name, conn)
                    self.duckdb_conn.register(table_name, df)

                    self.metadata[table_name] = DataFrameMetadata(
                        name=table_name,
                        columns=df.columns.tolist(),
                        sample_values={col: df[col].dropna().sample(min(5, len(df))).tolist()
                                       for col in df.columns},
                        cardinality={col: df[col].nunique() for col in df.columns},
                        total_rows=len(df)
                    )
                    self.logger.info(f"Loaded existing table '{table_name}' from SQLite")

        except Exception as e:
            self.logger.error(f"Error loading existing tables: {str(e)}")

    def add_dataframe(self, name: str, df: pd.DataFrame, description: str = "", source_type: str = "") -> None:
        """Add a DataFrame to both DuckDB and SQLite with metadata."""
        try:
            name = "".join(c if c.isalnum() else "_" for c in name)
            df.to_sql(name, self.sql_engine, if_exists='replace', index=False)
            self.duckdb_conn.register(name, df)

            self.metadata[name] = DataFrameMetadata(
                name=name,
                columns=df.columns.tolist(),
                sample_values={col: df[col].dropna().sample(min(5, len(df))).tolist()
                               for col in df.columns},
                cardinality={col: df[col].nunique() for col in df.columns},
                total_rows=len(df),
                description=description,
                source_type=source_type,
                last_updated=datetime.now()
            )
            self.logger.info(f"Successfully added DataFrame '{name}'")

        except Exception as e:
            self.logger.error(f"Error adding DataFrame '{name}': {str(e)}")
            raise


class QueryGenerator:
    """Generates SQL queries based on natural language input."""

    def __init__(self, df_manager: DataFrameManager, api_key: str):
        self.df_manager = df_manager
        self.openai_client = openai.Client(api_key=api_key)

    def generate_query(self, user_question: str) -> Tuple[str, str]:
        """Generate SQL query and summary prompt from user question."""
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

        # Extract the SQL query from the response and clean it
        raw_sql_query = response.choices[0].message.content
        sql_query = self._sanitize_sql_query(raw_sql_query)

        # Prepare a summary prompt
        summary_prompt = f"Summarize the following data in a concise, business-friendly way: [QUERY_RESULT]"

        return sql_query, summary_prompt

    def _sanitize_sql_query(self, raw_query: str) -> str:
        """Sanitize the generated SQL query by removing markdown formatting."""
        # Remove backticks and strip unnecessary whitespace
        sanitized_query = raw_query.replace("```sql", "").replace("```", "").strip()
        return sanitized_query

    def _create_context(self) -> str:
        """Create enhanced context string from DataFrame metadata."""
        context = []
        for name, meta in self.df_manager.metadata.items():
            table_info = [
                f"Table: {name}",
                f"Total Rows: {meta.total_rows}",
                "Columns:"
            ]
            for col in meta.columns:
                table_info.append(f"  - {col}")
            context.append("\n".join(table_info))
        return "\n\n".join(context)


class ChatBot:
    """Main chatbot class that handles user interactions."""

    def __init__(self, df_manager: DataFrameManager, query_generator: QueryGenerator):
        self.df_manager = df_manager
        self.query_generator = query_generator

    def process_question(self, question: str) -> Dict[str, Any]:
        """Process user question and return results with summary."""
        try:
            sql_query, summary_prompt = self.query_generator.generate_query(question)
            result_df = self.df_manager.duckdb_conn.execute(sql_query).fetchdf()

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
            return {"error": str(e)}


def setup_chatbot(api_key: str) -> ChatBot:
    """Setup and initialize the chatbot with the provided API key."""
    df_manager = DataFrameManager()
    query_generator = QueryGenerator(df_manager, api_key)
    return ChatBot(df_manager, query_generator)
