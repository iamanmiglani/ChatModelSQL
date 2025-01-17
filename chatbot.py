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
import re

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
    """Generates SQL queries based on natural language input with enhanced join support."""

    def __init__(self, df_manager: DataFrameManager, api_key: str):
        self.df_manager = df_manager
        self.openai_client = openai.Client(api_key=api_key)

    def generate_query(self, user_question: str) -> Tuple[str, str]:
        """Generate SQL query and summary prompt from user question."""
        context = self._create_enhanced_context()

        system_prompt = f"""You are a SQL expert. Generate a SQL query based on the following context and question. 
        The tables are stored in a DuckDB database. Only return the SQL query, nothing else.
        
        Important guidelines for generating queries:
        1. When joining tables, explicitly specify the join type (LEFT JOIN, RIGHT JOIN, INNER JOIN, FULL OUTER JOIN)
        2. Always use table aliases for clarity (e.g., 't1', 't2', etc.)
        3. Specify the full table.column name in SELECT statements to avoid ambiguity
        4. Include appropriate JOIN conditions based on the relationships described in the metadata
        
        Available tables and their metadata with relationships:
        {context}
        
        Example of a proper join query:
        SELECT t1.column1, t2.column2 
        FROM table1 t1 
        LEFT JOIN table2 t2 ON t1.id = t2.table1_id
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]

        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",  # Using GPT-4 for better query generation
            messages=messages,
            temperature=0
        )

        raw_sql_query = response.choices[0].message.content
        sql_query = self._sanitize_sql_query(raw_sql_query)

        # Validate the query structure
        if not self._validate_query(sql_query):
            raise ValueError("Generated query does not meet the required standards")

        summary_prompt = f"Summarize the following data in a concise, business-friendly way: [QUERY_RESULT]"
        return sql_query, summary_prompt

    def _create_enhanced_context(self) -> str:
        """Create enhanced context string from DataFrame metadata with relationship information."""
        context = []
        
        # First pass: Basic table information
        for name, meta in self.df_manager.metadata.items():
            table_info = [
                f"\nTable: {name}",
                f"Description: {meta.description}" if meta.description else "",
                f"Total Rows: {meta.total_rows}",
                "Columns:"
            ]
            
            # Enhanced column information with data types and relationships
            for col in meta.columns:
                sample_values = meta.sample_values.get(col, [])
                cardinality = meta.cardinality.get(col, 0)
                
                # Infer potential relationships based on column names and patterns
                relationship_info = self._infer_relationships(name, col, self.df_manager.metadata)
                
                col_info = f"  - {col}"
                if relationship_info:
                    col_info += f" (Potential join key: {relationship_info})"
                if cardinality:
                    col_info += f" (Unique values: {cardinality})"
                if sample_values:
                    col_info += f" (Sample values: {', '.join(str(v) for v in sample_values[:3])})"
                
                table_info.append(col_info)
            
            context.append("\n".join(filter(None, table_info)))

        return "\n\n".join(context)

    def _infer_relationships(self, table_name: str, column_name: str, metadata: Dict[str, Any]) -> str:
        """Infer potential relationships between tables based on column names."""
        relationship_patterns = [
            (r'(\w+)_id$', '{}_id matches table {}'),
            (r'id_(\w+)$', 'id_{} matches table {}'),
            (r'(\w+)_key$', '{}_key might relate to table {}'),
            (r'fk_(\w+)$', 'foreign key might relate to table {}')
        ]

        for pattern, message in relationship_patterns:
            match = re.match(pattern, column_name.lower())
            if match:
                related_table = match.group(1)
                # Check if the inferred table exists in our metadata
                for meta_table_name in metadata.keys():
                    if related_table in meta_table_name.lower():
                        return message.format(related_table, meta_table_name)
        return ""

    def _validate_query(self, query: str) -> bool:
        """Validate the generated query for proper join syntax and table references."""
        query_lower = query.lower()
        
        # Check for proper table aliases in joins
        if "join" in query_lower and not re.search(r'(\w+)\s+(?:as\s+)?[a-z][0-9a-z]*\s+(?:left|right|inner|full)', query_lower, re.IGNORECASE):
            return False

        # Check for fully qualified column names in SELECT
        if not re.search(r'select\s+(?:[a-z][0-9a-z]*\.)', query_lower, re.IGNORECASE):
            return False

        # Ensure JOIN conditions are present when using JOIN
        if "join" in query_lower and not re.search(r'join\s+\w+\s+(?:as\s+)?[a-z][0-9a-z]*\s+on\s+', query_lower, re.IGNORECASE):
            return False

        return True

    def _sanitize_sql_query(self, raw_query: str) -> str:
        """Sanitize and validate the generated SQL query."""
        # Remove markdown code blocks
        raw_query = raw_query.replace("```sql", "").replace("```", "").strip()

        # Use regex to find the first SQL-like statement
        sql_pattern = re.compile(r"(SELECT|WITH)\s+", re.IGNORECASE)
        match = sql_pattern.search(raw_query)

        if match:
            sanitized_query = raw_query[match.start():].strip()
            # Remove any trailing statements or comments
            sanitized_query = re.split(r';\s*--|\s*--|\s*;', sanitized_query)[0].strip()
            return sanitized_query
        else:
            raise ValueError("No valid SQL query found in the response.")


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
                model="gpt-3.5-turbo",
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
