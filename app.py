import streamlit as st

from src.data.loader import load_csv
from src.analysis.profiler import (
    get_dataset_overview,
    get_column_profile,
    get_numerical_statistics,
)
from src.analysis.quality import run_quality_checks
from src.analysis.eda import get_eda_kind, histogram, box_plot
from src.analysis.eda import (
    get_eda_kind,
    histogram,
    box_plot,
    category_counts,
    bar_chart,
    correlation_matrix,
    correlation_heatmap,
    scatter_plot,
    TOP_N_CATEGORIES,
)
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
    # ---------- EDA ----------
    st.header("4. Exploratory data analysis")

    numeric_cols = [c for c in df.columns if get_eda_kind(df[c]) == "numerical"]

    if not numeric_cols:
        st.write("This dataset has no numerical columns to chart.")
    else:
        st.subheader("Numerical columns")
        selected = st.selectbox("Choose a numerical column", numeric_cols)

        left, right = st.columns(2)
        left.plotly_chart(histogram(df, selected), use_container_width=True)
        right.plotly_chart(box_plot(df, selected), use_container_width=True)

        st.dataframe(
            get_numerical_statistics(df, [selected]), use_container_width=True
        )

    st.subheader("Categorical columns")
    categorical_cols = [c for c in df.columns if get_eda_kind(df[c]) == "categorical"]

    if not categorical_cols:
        st.write("This dataset has no categorical columns to chart.")
    else:
        chosen = st.selectbox("Choose a categorical column", categorical_cols)
        table = category_counts(df, chosen)

        left, right = st.columns(2)
        left.plotly_chart(bar_chart(table, chosen), use_container_width=True)
        right.dataframe(table, use_container_width=True)

        if df[chosen].nunique() > TOP_N_CATEGORIES:
            st.caption(
                f"Showing the {TOP_N_CATEGORIES} most frequent of "
                f"{df[chosen].nunique()} distinct values."
            )

    st.subheader("Correlations")
    if len(numeric_cols) < 2:
        st.write("At least two numerical columns are needed for correlations.")
    else:
        corr = correlation_matrix(df, numeric_cols)
        st.plotly_chart(correlation_heatmap(corr), use_container_width=True)
        st.caption(
            "Correlation measures linear association between two columns. "
            "It does not show that one causes the other."
        )

        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X axis", numeric_cols, index=0)
        y_col = c2.selectbox("Y axis", numeric_cols, index=1)
        st.plotly_chart(scatter_plot(df, x_col, y_col), use_container_width=True)

else:
    st.info("Upload a CSV dataset to begin.")