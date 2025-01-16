# Import libraries
import streamlit as st
import pandas as pd
from chatbot import setup_chatbot
from sqlalchemy import create_engine, inspect
import tempfile
import os
import plotly.express as px
from chats import ChartCodeGenerator  # Assuming chats.py contains ChartCodeGenerator

class StreamlitChatBot:
    def __init__(self):
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None
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
            if st.session_state.openai_api_key and st.session_state.chatbot is None:
                st.session_state.chatbot = setup_chatbot(st.session_state.openai_api_key)
                st.success("API Key Set and Chatbot Initialized!")

            st.header("Data Management")
            
            # Upload file
            uploaded_file = st.file_uploader("Upload a Data File", type=["csv", "xlsx", "xls", "db"])
            if uploaded_file:
                self.handle_file_upload(uploaded_file)

            # Display available tables
            if st.session_state.uploaded_tables:
                st.subheader("Available Tables")
                for table_name in st.session_state.uploaded_tables:
                    st.write(f"📊 {table_name}")

    def handle_file_upload(self, uploaded_file):
        """Handle file uploads and add tables to the chatbot."""
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
                
                # Add DataFrame to chatbot's database
                if st.session_state.chatbot:
                    st.session_state.chatbot.df_manager.add_dataframe(
                        name=table_name,
                        df=df,
                        description=f"Uploaded file: {uploaded_file.name}"
                    )
                    st.success(f"Table '{table_name}' added successfully!")
                    st.session_state.uploaded_tables.append(table_name)
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
        if user_input and st.session_state.chatbot:
            response = st.session_state.chatbot.process_question(user_input)
            if "error" in response:
                st.error(f"Error: {response['error']}")
            else:
                st.write("### Query:")
                st.code(response["query"])
                st.write("### Results:")
                result_df = pd.DataFrame(response["data"])
                st.dataframe(result_df)
                st.write("### Summary:")
                st.info(response["summary"])
                
                # Store the query results for visualization
                st.session_state.query_results = result_df

                # Add a Visualize button
                if st.button("Visualize"):
                    st.session_state.show_visualization = True

        # Visualization Section
        if st.session_state.show_visualization and st.session_state.query_results is not None:
            self.render_visualization_tab(st.session_state.query_results)

    def render_visualization_tab(self, df: pd.DataFrame):
        st.header("Visualize Your Data")

        # Chart customization options
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram"])
        x_axis = st.selectbox("Select X-Axis", df.columns)
        y_axis = st.selectbox("Select Y-Axis", df.columns if chart_type != "Pie Chart" else [""])
        color = st.color_picker("Pick a Color", "#636EFA")

        # Generate chart
        if st.button("Generate Chart"):
            chart_generator = ChartCodeGenerator(api_key=st.session_state.openai_api_key)
            chart_codes = chart_generator.generate_chart_code(df)
            code = chart_codes.get(chart_type, "")

            try:
                # Execute and display the generated chart
                exec_globals = {}
                exec(code, {"pd": pd, "px": px}, exec_globals)
                fig = exec_globals.get("fig", None)
                if fig:
                    fig.update_traces(marker=dict(color=color))  # Apply custom color
                    st.plotly_chart(fig)
                else:
                    st.error("Could not generate the chart.")
            except Exception as e:
                st.error(f"Error generating chart: {e}")


def main():
    app = StreamlitChatBot()
    app.setup_page()
    app.render_sidebar()
    app.render_chat_interface()


if __name__ == "__main__":
    main()
