import os
import pandas as pd
import duckdb
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from openai import OpenAI  # Updated import
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
        # Updated OpenAI client initialization
        self.openai_client = OpenAI(api_key=api_key)

    def generate_query(self, user_question: str) -> Tuple[str, str]:
        """Generate SQL query and summary prompt from user question."""
        context = self._create_enhanced_context()

        system_prompt = f"""You are a SQL expert. Generate a SQL query based on the following context and question. 
        The tables are stored in a DuckDB database. Only return the SQL query, nothing else.
        
        Guidelines:
        1. For simple queries on a single table, use straightforward SELECT statements
        2. When joining tables:
           - Use explicit join types (LEFT JOIN, RIGHT JOIN, INNER JOIN)
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
            
            # Only validate if query contains joins
            if 'join' in sql_query.lower() and not self._validate_query(sql_query):
                raise ValueError("Query validation failed - please try rephrasing your question")

            summary_prompt = f"Summarize the following data in a concise, business-friendly way: [QUERY_RESULT]"
            return sql_query, summary_prompt

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

    def _validate_query(self, query: str) -> bool:
        """Validate the generated query structure."""
        query_lower = query.lower()
        
        # Skip validation for simple queries
        if "join" not in query_lower:
            return True
            
        try:
            # Basic SQL structure check
            if not re.search(r'select .+ from .+', query_lower, re.IGNORECASE):
                return False
                
            # Validate join syntax
            if "join" in query_lower:
                join_pattern = r'join\s+(\w+)(?:\s+(?:as\s+)?(\w+))?\s+on\s+'
                if not re.search(join_pattern, query_lower, re.IGNORECASE):
                    return False
                    
            return True
            
        except Exception:
            # If validation fails, assume query is valid
            return True

    def _sanitize_sql_query(self, raw_query: str) -> str:
        """Clean and validate the SQL query."""
        # Remove markdown formatting
        clean_query = raw_query.replace("```sql", "").replace("```", "").strip()
        
        # Extract the SQL statement
        sql_pattern = re.compile(r"(SELECT|WITH)\s+", re.IGNORECASE)
        match = sql_pattern.search(clean_query)
        
        if not match:
            raise ValueError("No valid SQL query found in the response")
            
        query = clean_query[match.start():].strip()
        return re.split(r';\s*--|\s*--|\s*;', query)[0].strip()


class ChatBot:
    """Main chatbot class that handles user interactions."""

    def __init__(self, df_manager: DataFrameManager, query_generator: QueryGenerator):
        self.df_manager = df_manager
        self.query_generator = query_generator
        self.logger = logging.getLogger(__name__)

    def process_question(self, question: str) -> Dict[str, Any]:
        """Process user question and return formatted results."""
        try:
            # Generate and execute query
            sql_query, summary_prompt = self.query_generator.generate_query(question)
            result_df = self.df_manager.duckdb_conn.execute(sql_query).fetchdf()

            # Generate result summary
            summary_messages = [
                {"role": "system", "content": "You are a data analyst. Provide a clear, concise summary of the data results."},
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
            self.logger.error(f"Error processing question: {str(e)}")
            return {"error": str(e)}


def setup_chatbot(api_key: str) -> ChatBot:
    """Initialize and configure the chatbot."""
    try:
        # Validate API key by creating a test client
        test_client = OpenAI(api_key=api_key)
        # Test the API key with a minimal request
        test_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        
        df_manager = DataFrameManager()
        query_generator = QueryGenerator(df_manager, api_key)
        return ChatBot(df_manager, query_generator)
    except Exception as e:
        raise ValueError(f"Failed to initialize chatbot with provided API key: {str(e)}")
