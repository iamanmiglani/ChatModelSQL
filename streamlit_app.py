import streamlit as st
import pandas as pd
import plotly.express as px
from charts import ChartCodeGenerator


class StreamlitChatBot:
    def __init__(self):
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = None
        if 'query_results' not in st.session_state:
            st.session_state.query_results = None
        if 'show_visualization' not in st.session_state:
            st.session_state.show_visualization = False

    def setup_page(self):
        st.set_page_config(page_title="AI Chatbot with Visualization", layout="wide")
        st.title("AI Chatbot with Visualization")

    def render_sidebar(self):
        with st.sidebar:
            st.header("Settings")
            
            # Input OpenAI API key
            st.session_state.openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
            if st.session_state.openai_api_key and st.session_state.chatbot is None:
                st.success("API Key Set!")

    def render_chat_interface(self):
        st.header("Chat Interface")
        if not st.session_state.openai_api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
            return

        # Query results simulation (replace with actual chatbot integration)
        if st.button("Simulate Query Results"):
            st.session_state.query_results = pd.DataFrame({
                "Category": ["A", "B", "C", "D"],
                "Values": [10, 20, 30, 40],
                "Counts": [100, 200, 300, 400]
            })

        if st.session_state.query_results is not None:
            st.write("### Query Results")
            st.dataframe(st.session_state.query_results)

            # Visualization button
            if st.button("Visualize"):
                st.session_state.show_visualization = True

        if st.session_state.show_visualization:
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

            if st.checkbox("Show Generated Code"):
                st.code(code)

            try:
                # Execute and display the generated chart
                exec_globals = {}
                exec(code, {"pd": pd, "px": px}, exec_globals)
                fig = exec_globals.get("fig", None)
                if fig:
                    fig.update_traces(marker=dict(color=color))  # Apply custom color
                    st.plotly_chart(fig)
                else:
                    st.error("Could not generate the chart. No figure returned.")
            except Exception as e:
                st.error(f"Error generating chart: {e}")


def main():
    app = StreamlitChatBot()
    app.setup_page()
    app.render_sidebar()
    app.render_chat_interface()


if __name__ == "__main__":
    main()
