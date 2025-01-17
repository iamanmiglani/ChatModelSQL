## Working
import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Optional
from openai import OpenAI

class ChartCodeGenerator:
    """Generates and executes Plotly charts based on DataFrame content."""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Define standard chart templates
        self.chart_templates = {
            "Bar Chart": self._create_bar_chart,
            "Line Chart": self._create_line_chart,
            "Scatter Plot": self._create_scatter_plot,
            "Pie Chart": self._create_pie_chart,
            "Histogram": self._create_histogram
        }

    def generate_chart(self, df: pd.DataFrame, chart_type: str, x_column: str, 
                      y_column: Optional[str] = None, color: str = "#636EFA") -> go.Figure:
        """
        Generate a Plotly chart directly instead of generating code.
        
        Args:
            df: Input DataFrame
            chart_type: Type of chart to generate
            x_column: Column to use for x-axis
            y_column: Column to use for y-axis (optional)
            color: Color to use for the chart
            
        Returns:
            Plotly figure object
        """
        if chart_type not in self.chart_templates:
            raise ValueError(f"Unsupported chart type: {chart_type}")
            
        try:
            fig = self.chart_templates[chart_type](df, x_column, y_column, color)
            self.logger.info(f"Successfully generated {chart_type}")
            return fig
        except Exception as e:
            self.logger.error(f"Error generating {chart_type}: {str(e)}")
            raise

    def _create_bar_chart(self, df: pd.DataFrame, x: str, y: str, color: str) -> go.Figure:
        fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
        fig.update_traces(marker_color=color)
        return fig

    def _create_line_chart(self, df: pd.DataFrame, x: str, y: str, color: str) -> go.Figure:
        fig = px.line(df, x=x, y=y, title=f"{y} over {x}")
        fig.update_traces(line_color=color)
        return fig

    def _create_scatter_plot(self, df: pd.DataFrame, x: str, y: str, color: str) -> go.Figure:
        fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
        fig.update_traces(marker_color=color)
        return fig

    def _create_pie_chart(self, df: pd.DataFrame, names: str, values: str, color: str) -> go.Figure:
        fig = px.pie(df, names=names, values=values, title=f"Distribution of {values}")
        # For pie charts, we'll use a color sequence based on the main color
        return fig

    def _create_histogram(self, df: pd.DataFrame, x: str, y: str = None, color: str = None) -> go.Figure:
        fig = px.histogram(df, x=x, title=f"Distribution of {x}")
        fig.update_traces(marker_color=color)
        return fig

    def suggest_chart_type(self, df: pd.DataFrame, x_column: str, y_column: Optional[str] = None) -> str:
        """Suggest the most appropriate chart type based on the data."""
        try:
            prompt = f"""
            Suggest the best chart type for visualizing data with these characteristics:
            X column ({x_column}): {df[x_column].dtype}
            Y column ({y_column}): {df[y_column].dtype if y_column else 'None'}
            Number of unique X values: {df[x_column].nunique()}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            
            suggestion = response.choices[0].message.content
            return suggestion
            
        except Exception as e:
            self.logger.error(f"Error suggesting chart type: {str(e)}")
            return "Bar Chart"  # Default to bar chart if suggestion fails
