import plotly.express as px

from src.analysis.profiler import classify_column

# A "numerical" column with this few distinct values behaves like a category
# (for example a 0/1 flag), so charts should treat it as one.
BINARY_MAX_UNIQUE = 2

# Bar charts show only the most frequent categories.
TOP_N_CATEGORIES = 15

def get_eda_kind(series):
    """Column kind for charting: like classify_column, but 0/1-style
    numerical columns are treated as categorical."""
    kind = classify_column(series)
    if kind == "numerical" and series.nunique() <= BINARY_MAX_UNIQUE:
        return "categorical"
    return kind


def histogram(df, column):
    """Histogram of a numerical column."""
    fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
    fig.update_layout(bargap=0.05)
    return fig


def box_plot(df, column):
    """Box plot of a numerical column (outliers shown as dots)."""
    return px.box(df, y=column, points="outliers", title=f"Box plot of {column}")
def category_counts(df, column, top_n=TOP_N_CATEGORIES):
    """Counts and percentages of the most frequent categories in a column."""
    counts = df[column].value_counts().head(top_n)
    total = int(df[column].notna().sum())

    table = counts.rename_axis("Category").reset_index(name="Count")
    table["Percent"] = (table["Count"] / total * 100).round(2)
    table["Category"] = table["Category"].astype(str)
    return table


def bar_chart(table, column):
    """Bar chart from a category_counts table."""
    fig = px.bar(
        table,
        x="Category",
        y="Count",
        hover_data=["Percent"],
        title=f"Most frequent values of {column}",
    )
    fig.update_xaxes(type="category")
    return fig


def correlation_matrix(df, numerical_columns):
    """Pearson correlation between every pair of numerical columns."""
    return df[numerical_columns].corr().round(2)


def correlation_heatmap(corr):
    """Heatmap of a correlation matrix (red = positive, blue = negative)."""
    return px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        title="Correlation between numerical columns",
    )


def scatter_plot(df, x, y):
    """Scatter plot of two numerical columns."""
    return px.scatter(df, x=x, y=y, opacity=0.5, title=f"{y} vs {x}")