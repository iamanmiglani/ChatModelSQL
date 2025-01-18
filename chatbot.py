# Base line 2 (tests required)
import os
import pandas as pd
import duckdb
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI
from datetime import datetime
import json
import logging
from sqlalchemy import create_engine, text
import re
from join import JoinHandler  # Import the JoinHandler for table relationships

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

    def delete_table(self, table_name: str) -> None:
        """Delete a table permanently from SQLite and DuckDB."""
        try:
            with self.sql_engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            self.duckdb_conn.unregister(table_name)
            self.metadata.pop(table_name, None)
            self.logger.info(f"Successfully deleted table '{table_name}' from the database.")
        except Exception as e:
            self.logger.error(f"Error deleting table '{table_name}': {str(e)}")
            raise


class QueryGenerator:
    """Generates SQL queries based on natural language input with enhanced join support."""

    def __init__(self, df_manager: DataFrameManager, api_key: str):
        self.df_manager = df_manager
        self.openai_client = OpenAI(api_key=api_key)
        self.join_handler = JoinHandler(
            {name: pd.DataFrame.from_dict(meta.sample_values) for name, meta in df_manager.metadata.items()}
        )  # Initialize JoinHandler with metadata

    def generate_query(self, user_question: str) -> Tuple[str, str]:
        """Generate SQL query and summary prompt from user question."""
        context = self._create_enhanced_context()

        # Determine if joins are required
        join_details = self.join_handler.get_join_details(user_question)
        relevant_tables = join_details["relevant_tables"]
        joins = join_details["joins"]

        # If only one table is relevant, generate a simple query
        if len(relevant_tables) == 1:
            table_name = relevant_tables[0]
            sql_query = f"SELECT * FROM {table_name}"
            return sql_query, self._summarize_output_table(table_name)

        # If joins are required, generate a join query
        if joins:
            join_query = self.join_handler.generate_join_query(user_question)
            output_table_summary = self._summarize_output_table_from_query(join_query)
            return join_query, output_table_summary

        # Fallback to OpenAI-based generation if no relevant tables or joins are found
        system_prompt = f"""You are a SQL expert. Generate a SQL query based on the following context and question. 
        The tables are stored in a DuckDB database. Only return the SQL query, nothing else.

        Guidelines:
        1. For simple queries on a single table, use straightforward SELECT statements
        2. When joining tables:
           - Use explicit join types (LEFT JOIN, RIGHT JOIN, FULL JOIN) as required
           - Identify primary and foreign key relationships automatically
           - Match columns that appear to be related (e.g., id fields, matching names)
           - Use clear ON conditions
        3. Include WHERE, GROUP BY, or HAVING clauses as needed
        4. Use appropriate aggregations (SUM, COUNT, AVG) when needed
        5. Handle NULL values appropriately
        
        Available tables and their metadata:
        {context}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )

            raw_sql_query = response.choices[0].message.content
            sql_query = self._sanitize_sql_query(raw_sql_query)
            output_table_summary = self._summarize_output_table_from_query(sql_query)

            return sql_query, output_table_summary

        except Exception as e:
            raise ValueError(f"Error generating query: {str(e)}")

    def _create_enhanced_context(self) -> str:
        """Create enhanced context string with table relationships."""
        context_parts = []

        for name, meta in self.df_manager.metadata.items():
            table_info = [
                f"\nTable: {name}",
                f"Description: {meta.description}" if meta.description else "",
                f"Total Rows: {meta.total_rows}",
                "Columns:"
            ]

            for col in meta.columns:
                sample_values = meta.sample_values.get(col, [])
                cardinality = meta.cardinality.get(col, 0)
                data_type = self._infer_column_type(sample_values)

                col_info = [f"  - {col} ({data_type})"]
                if cardinality:
                    col_info.append(f"Unique values: {cardinality}")
                if sample_values:
                    col_info.append(f"Examples: {', '.join(str(v) for v in sample_values[:3])}")

                table_info.append(" | ".join(col_info))

            context_parts.append("\n".join(filter(None, table_info)))

        return "\n\n".join(context_parts)

    def _summarize_output_table(self, table_name: str) -> str:
        """Summarize the output table based on its metadata."""
        if table_name not in self.df_manager.metadata:
            return f"Table '{table_name}' not found in metadata."

        meta = self.df_manager.metadata[table_name]
        summary = [
            f"Summary of table '{meta.name}':",
            f"- Total Rows: {meta.total_rows}",
            "- Columns:"
        ]

        for col in meta.columns:
            cardinality = meta.cardinality.get(col, 0)
            examples = ", ".join(map(str, meta.sample_values.get(col, [])[:3]))
            summary.append(f"  - {col}: {cardinality} unique values, examples: {examples}")

        return "\n".join(summary)

    def _summarize_output_table_from_query(self, query: str) -> str:
        """Execute the query and summarize the output table."""
        try:
            df = self.df_manager.duckdb_conn.execute(query).fetchdf()
            total_rows = len(df)
            summary = [
                f"Summary of query output:",
                f"- Total Rows: {total_rows}",
                "- Columns:"
            ]

            for col in df.columns:
                unique_values = df[col].nunique()
                examples = ", ".join(map(str, df[col].dropna().sample(min(3, total_rows)).tolist()))
                summary.append(f"  - {col}: {unique_values} unique values, examples: {examples}")

            return "\n".join(summary)
        except Exception as e:
            return f"Error summarizing output table: {str(e)}"

    def _infer_column_type(self, values: List[Any]) -> str:
        """Infer the data type of a column from its values."""
        if not values:
            return "unknown"
        sample = values[0]
        if isinstance(sample, (int, np.integer)):
            return "integer"
        elif isinstance(sample, (float, np.floating)):
            return "numeric"
        elif isinstance(sample, (datetime, np.datetime64)):
            return "datetime"
        return "text"

    def _sanitize_sql_query(self, raw_query: str) -> str:
        """Clean and validate the SQL query."""
        clean_query = raw_query.replace("```sql", "").replace("```", "").strip()

        sql_pattern = re.compile(r"(SELECT|WITH)\s+", re.IGNORECASE)
        match = sql_pattern.search(clean_query)

        if not match:
            raise ValueError("No valid SQL query found in the response")

        query = clean_query[match.start():].strip()
        return re.split(r';\s*--|\s*--|\s*;', query)[0].strip()
