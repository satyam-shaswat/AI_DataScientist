import streamlit as st

from src.data.loader import load_csv
from src.analysis.profiler import (
    get_dataset_overview,
    get_column_profile,
    get_numerical_statistics,
)
from src.analysis.quality import run_quality_checks
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="📊",
    layout="wide",
)

st.title("AI Data Scientist")
st.caption("Intelligent Data Analysis & Machine Learning Platform")

st.header("1. Upload dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        st.session_state["df"] = df
        st.session_state["filename"] = uploaded_file.name
    except ValueError as e:
        st.error(f"Could not load this file: {e}")

if "df" in st.session_state:
    df = st.session_state["df"]

    st.success(f"Dataset loaded: {st.session_state['filename']}")

    converted = df.attrs.get("converted_columns", [])
    if converted:
        st.info(
            f"Converted text columns to numbers: {', '.join(converted)}. "
            "Blank entries were treated as missing values."
        )

    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])

    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # ---------- Dataset profile ----------
    st.header("2. Dataset profile")
    overview = get_dataset_overview(df)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Memory (MB)", overview["memory_mb"])
    m2.metric("Numerical", len(overview["numerical_columns"]))
    m3.metric("Categorical", len(overview["categorical_columns"]))
    m4.metric("Datetime", len(overview["datetime_columns"]))
    m5.metric("Potential IDs", len(overview["identifier_columns"]))

    if overview["identifier_columns"]:
        st.caption(
            "Potential identifier columns: "
            + ", ".join(overview["identifier_columns"])
        )

    st.subheader("Column profile")
    st.dataframe(get_column_profile(df), use_container_width=True)

    st.subheader("Numerical statistics")
    stats = get_numerical_statistics(df, overview["numerical_columns"])
    if stats.empty:
        st.write("This dataset has no numerical columns.")
    else:
        st.dataframe(stats, use_container_width=True)
        # ---------- Data quality ----------
    st.header("3. Data quality")
    findings = run_quality_checks(df)

    if not findings:
        st.success("No data-quality issues were detected by the current checks.")
    else:
        for f in findings:
            if f["severity"] == "warning":
                st.warning(f["message"])
            else:
                st.info(f["message"])
else:
    st.info("Upload a CSV dataset to begin.")