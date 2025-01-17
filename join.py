import pandas as pd
from typing import List, Dict, Tuple, Any
from collections import defaultdict

class JoinHandler:
    """Class to handle schema analysis and identify relationships between tables."""

    def __init__(self, table_metadata: Dict[str, pd.DataFrame]):
        self.table_metadata = table_metadata
        self.relationships = self._identify_relationships()

    def _identify_relationships(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """Identify potential relationships between tables based on column names and data."""
        relationships = defaultdict(list)

        table_names = list(self.table_metadata.keys())
        for i, table_a in enumerate(table_names):
            for j, table_b in enumerate(table_names):
                if i >= j:
                    continue

                df_a = self.table_metadata[table_a]
                df_b = self.table_metadata[table_b]

                for col_a in df_a.columns:
                    for col_b in df_b.columns:
                        if col_a == col_b or col_a.lower() == col_b.lower():
                            relationship_type = self._determine_relationship_type(
                                df_a[col_a], df_b[col_b]
                            )
                            relationships[(table_a, table_b)].append(
                                (col_a, col_b, relationship_type)
                            )

        return relationships

    def _determine_relationship_type(self, col_a: pd.Series, col_b: pd.Series) -> str:
        """Determine the relationship type between two columns (e.g., one-to-one, one-to-many)."""
        if col_a.nunique() == len(col_a) and col_b.nunique() == len(col_b):
            return "one-to-one"
        elif col_a.nunique() == len(col_a):
            return "one-to-many"
        elif col_b.nunique() == len(col_b):
            return "many-to-one"
        return "many-to-many"

    def get_join_details(self, user_question: str) -> Dict[str, Any]:
        """Determine which tables and columns to join based on the user question."""
        relevant_tables = self._identify_relevant_tables(user_question)
        joins = []

        if len(relevant_tables) > 1:
            for i in range(len(relevant_tables)):
                for j in range(i + 1, len(relevant_tables)):
                    table_a = relevant_tables[i]
                    table_b = relevant_tables[j]

                    if (table_a, table_b) in self.relationships:
                        joins.extend(self.relationships[(table_a, table_b)])

        return {
            "relevant_tables": relevant_tables,
            "joins": joins
        }

    def _identify_relevant_tables(self, user_question: str) -> List[str]:
        """Identify which tables are relevant to the user's question."""
        keywords = user_question.lower().split()
        relevant_tables = []

        for table_name, df in self.table_metadata.items():
            if any(keyword in table_name.lower() for keyword in keywords):
                relevant_tables.append(table_name)

            # Check column names for relevance
            for col in df.columns:
                if any(keyword in col.lower() for keyword in keywords):
                    relevant_tables.append(table_name)
                    break

        return list(set(relevant_tables))

    def generate_join_query(self, user_question: str) -> str:
        """Generate a SQL query with appropriate joins based on the user question."""
        join_details = self.get_join_details(user_question)
        relevant_tables = join_details["relevant_tables"]
        joins = join_details["joins"]

        if not relevant_tables:
            return "No relevant tables identified for the question."

        # Start building the query
        base_table = relevant_tables[0]
        query = f"SELECT * FROM {base_table}"

        # Track joined tables to avoid duplicate joins
        joined_tables = set()
        for table_a, table_b, col_a, col_b, relationship_type in [
            (a, b, c[0], c[1], c[2]) for a, b in self.relationships for c in self.relationships[(a, b)]
        ]:
            if table_a in relevant_tables and table_b in relevant_tables and table_b not in joined_tables:
                query += f"\nLEFT JOIN {table_b} ON {table_a}.{col_a} = {table_b}.{col_b}"
                joined_tables.add(table_b)

        return query

# Example Usage
if __name__ == "__main__":
    # Example DataFrames (replace with real data)
    df1 = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"]
    })

    df2 = pd.DataFrame({
        "id": [1, 2, 4],
        "age": [25, 30, 35]
    })

    df3 = pd.DataFrame({
        "user_id": [1, 2, 3],
        "country": ["US", "UK", "CA"]
    })

    table_metadata = {
        "users": df1,
        "details": df2,
        "locations": df3
    }

    join_handler = JoinHandler(table_metadata)
    user_question = "What are the names, ages, and countries of users?"

    sql_query = join_handler.generate_join_query(user_question)
    print("Generated SQL Query:")
    print(sql_query)
