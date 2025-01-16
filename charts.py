import openai
import pandas as pd
import logging


class ChartCodeGenerator:
    """Generates Python code for visualizing data using Plotly."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_chart_code(self, df: pd.DataFrame) -> dict:
        """
        Generate Python code for all possible chart types based on the given DataFrame.
        Returns a dictionary with chart types as keys and corresponding Python code as values.
        """
        columns = df.columns.tolist()
        prompt = f"""
        You are a Python data visualization expert. Generate Python code for visualizing data using Plotly.
        Use the following DataFrame columns:
        {columns}

        Generate Python code for:
        1. Bar chart
        2. Line chart
        3. Scatter plot
        4. Pie chart
        5. Histogram

        Use the column names dynamically for axes. Include Plotly imports in the code.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert Python data visualization assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5
            )

            code = response['choices'][0]['message']['content']
            self.logger.info("OpenAI Response: %s", code)
            return self._split_code_into_charts(code)

        except Exception as e:
            self.logger.error(f"Error in OpenAI API call: {e}")
            return {}

    def _split_code_into_charts(self, code: str) -> dict:
        """Splits a single block of code into multiple chart-specific code snippets."""
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
