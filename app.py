import streamlit as st
from src.data.loader import load_csv


st.set_page_config(
    page_title="AI Data Scientist",
    page_icon=":robot_face:",
    layout="wide",
)

st.title("AI Data Scientist")
st.caption("Intelligent Data Analysis & Machine Learning Platform")
st.header("Upload Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = load_csv(uploaded_file)
        st.session_state["data"] = data
        st.session_state["filename"] = uploaded_file.name
        st.success(f"Dataset loaded successfully: {uploaded_file.name}")

        col1, col2 = st.columns(2)
        col1.metric("Rows", f"{data.shape[0]:}")
        col2.metric("Columns", f"{data.shape[1]:}")

        st.subheader("Preview")
        st.dataframe(data.head(10), use_container_width=True)

        st.subheader("Column names")
        st.write(list(data.columns))
    except ValueError as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Upload a CSV dataset to begin.")

