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
