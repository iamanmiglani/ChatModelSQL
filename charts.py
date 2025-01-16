import openai
import pandas as pd
from typing import List, Dict, Any


class ChartCodeGenerator:
    """Generates Python code for visualizing data using Plotly."""
    
    def __init__(self, api_key: str):
        self.openai_client = openai.Client(api_key=api_key)

    def generate_chart_code(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Generate Python code for all possible chart types based on the given DataFrame.
        Returns a dictionary with chart types as keys and corresponding Python code as values.
        """
        # Convert DataFrame metadata to a JSON-like structure
        columns = df.columns.tolist()
        metadata = {
            "columns": columns,
            "sample_data": df.head(5).to_dict(orient="records"),
        }

        # Prepare the prompt for the OpenAI API
        prompt = f"""
        You are a Python data visualization expert. Using Plotly, generate Python code for all possible chart types
        that can be created using the following DataFrame metadata:
        
        Metadata:
        {metadata}

        Generate code for the following chart types:
        - Bar chart
        - Line chart
        - Scatter plot
        - Pie chart
        - Histogram

        For each chart, ensure that:
        - The user can customize the x-axis, y-axis, and color.
        - Use sample data from the metadata to demonstrate the charts.

        Provide a separate Python code block for each chart type.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert Python data visualization assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5
        )

        # Process the response
        chart_code = response.choices[0].message.content

        # Split the code into separate chart types
        chart_blocks = self._split_chart_code(chart_code)
        return chart_blocks

    def _split_chart_code(self, code: str) -> Dict[str, str]:
        """
        Split the generated code into separate blocks for each chart type.
        """
        chart_blocks = {}
        current_chart = None
        chart_lines = []

        for line in code.splitlines():
            if line.startswith("#") and "chart" in line.lower():
                if current_chart:
                    chart_blocks[current_chart] = "\n".join(chart_lines)
                current_chart = line.strip("# ").strip()
                chart_lines = []
            elif current_chart:
                chart_lines.append(line)

        if current_chart:
            chart_blocks[current_chart] = "\n".join(chart_lines)

        return chart_blocks
