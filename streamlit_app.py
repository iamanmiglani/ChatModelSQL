import streamlit as st
import pandas as pd
from chatbot import DataFrameManager, QueryGenerator
from sqlalchemy import create_engine, inspect
import tempfile
import os
import plotly.express as px
from charts import ChartCodeGenerator

class StreamlitChatBot:
    def __init__(self):
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        if 'df_manager' not in st.session_state:
            st.session_state.df_manager = None
        if 'query_generator' not in st.session_state:
            st.session_state.query_generator = None
        if 'uploaded_tables' not in st.session_state:
            st.session_state.uploaded_tables = []
        if 'query_results' not in st.session_state:
            st.session_state.query_results = None
        if 'show_visualization' not in st.session_state:
            st.session_state.show_visualization = False

    def setup_page(self):
        st.set_page_config(page_title="AI Chatbot with Data Upload and Visualization", layout="wide")
        st.title("AI Chatbot with Data Management and Visualization")

    def render_sidebar(self):
        with st.sidebar:
            st.header("Settings")

            # Input OpenAI API key
            st.session_state.openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
            if st.session_state.openai_api_key and st.session_state.df_manager is None:
                st.session_state.df_manager = DataFrameManager()
                st.session_state.query_generator = QueryGenerator(
                    st.session_state.df_manager, st.session_state.openai_api_key
                )
                st.success("API Key Set and Chatbot Initialized!")

            st.header("Data Management")

            # Show available tables and delete option
            if st.session_state.df_manager:
                tables = list(st.session_state.df_manager.metadata.keys())
                if tables:
                    st.subheader("Available Tables")
                    st.caption("Double-click to delete a table.")
                    for table_name in tables:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"📊 {table_name}")
                        with col2:
                            if st.button("❌", key=f"delete_{table_name}"):
                                st.session_state.df_manager.delete_table(table_name)
                                st.success(f"Table '{table_name}' deleted successfully!")
                                st.session_state.uploaded_tables.remove(table_name)
                                st.query_params(refresh="true")

            # Upload file
            uploaded_file = st.file_uploader("Upload a Data File", type=["csv", "xlsx", "xls", "db"])
            if uploaded_file:
                self.handle_file_upload(uploaded_file)

    def handle_file_upload(self, uploaded_file):
        """Handle file uploads and add tables to the DataFrameManager."""
        file_type = uploaded_file.name.split('.')[-1].lower()
        table_name = st.text_input("Enter Table Name", value=uploaded_file.name.split('.')[0])

        if st.button("Add Table"):
            try:
                if file_type == "csv":
                    df = pd.read_csv(uploaded_file)
                elif file_type in ["xls", "xlsx"]:
                    df = pd.read_excel(uploaded_file)
                elif file_type == "db":
                    df = self.load_table_from_sqlite(uploaded_file)
                else:
                    st.error("Unsupported file type!")
                    return

                # Add DataFrame to the DataFrameManager
                if st.session_state.df_manager:
                    st.session_state.df_manager.add_dataframe(
                        name=table_name,
                        df=df,
                        description=f"Uploaded file: {uploaded_file.name}"
                    )
                    st.success(f"Table '{table_name}' added successfully!")
                    if table_name not in st.session_state.uploaded_tables:
                        st.session_state.uploaded_tables.append(table_name)
                    st.query_params(refresh="true")
                else:
                    st.warning("Please set the OpenAI API key first.")
            except Exception as e:
                st.error(f"Error adding table: {e}")

    def load_table_from_sqlite(self, uploaded_file):
        """Load a table from an uploaded SQLite database file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        engine = create_engine(f"sqlite:///{temp_path}")
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        if not table_names:
            st.error("No tables found in the SQLite database!")
            os.unlink(temp_path)
            return None

        selected_table = st.selectbox("Select a table to import", table_names)
        if selected_table:
            df = pd.read_sql_table(selected_table, engine)
            os.unlink(temp_path)
            return df

    def render_chat_interface(self):
        st.header("Chat Interface")
        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
            return

        user_input = st.text_input("Ask a question:")
        if user_input and st.session_state.query_generator:
            try:
                sql_query, summary_prompt = st.session_state.query_generator.generate_query(user_input)
                result_df = st.session_state.df_manager.duckdb_conn.execute(sql_query).fetchdf()

                st.write("### Query:")
                st.code(sql_query)
                st.write("### Results:")
                st.dataframe(result_df)
                st.write("### Summary:")
                st.info(summary_prompt)

                # Store the query results for visualization
                st.session_state.query_results = result_df

                # Add a Visualize button
                if st.button("Visualize"):
                    st.session_state.show_visualization = True

            except Exception as e:
                st.error(f"Error processing question: {str(e)}")

        # Visualization Section
        if st.session_state.show_visualization and st.session_state.query_results is not None:
            self.render_visualization_tab(st.session_state.query_results)

    def render_visualization_tab(self, df: pd.DataFrame):
        st.header("Visualize Your Data")

        # Initialize chart generator
        chart_generator = ChartCodeGenerator(api_key=st.session_state.openai_api_key)

        # Chart customization options
        chart_type = st.selectbox("Select Chart Type", list(chart_generator.chart_templates.keys()))

        # Get numeric columns for y-axis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

        # Select columns based on chart type
        x_axis = st.selectbox("Select X-Axis", df.columns)

        if chart_type != "Pie Chart" and chart_type != "Histogram":
            y_axis = st.selectbox("Select Y-Axis", numeric_cols)
        else:
            y_axis = None if chart_type == "Histogram" else x_axis

        color = st.color_picker("Pick a Color", "#636EFA")

        # Generate chart
        if st.button("Generate Chart"):
            try:
                fig = chart_generator.generate_chart(
                    df=df,
                    chart_type=chart_type,
                    x_column=x_axis,
                    y_column=y_axis,
                    color=color
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add chart suggestion
                suggestion = chart_generator.suggest_chart_type(df, x_axis, y_axis)
                st.info(f"💡 Suggested visualization: {suggestion}")

            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")


def main():
    app = StreamlitChatBot()
    app.setup_page()
    app.render_sidebar()
    app.render_chat_interface()


if __name__ == "__main__":
    main()
