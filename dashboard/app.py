
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sales Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "retail-orders.csv"
LR_PATH = BASE_DIR / "outputs" / "predictions.csv"
RF_PATH = BASE_DIR / "outputs" / "rf_predictions.csv"

# =========================================================
# ADVANCED DARK THEME
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #080e1b 0%, #101a2c 55%, #0b1423 100%);
    color: #edf4ff;
}

[data-testid="stSidebar"] {
    background: #0b1423;
    border-right: 1px solid #23334c;
}

[data-testid="stHeader"] {
    background: transparent;
}

.main-title {
    font-size: 35px;
    font-weight: 800;
    color: #f1f6ff;
    letter-spacing: -1px;
}

.subtitle {
    color: #8da4c2;
    font-size: 14px;
    margin-top: -12px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #eaf2ff;
    margin-top: 12px;
}

.metric-card {
    background: linear-gradient(145deg, #172943, #101d31);
    border: 1px solid #29415f;
    padding: 21px;
    border-radius: 15px;
    min-height: 115px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.18);
}

.metric-label {
    color: #9db2ce;
    font-size: 13px;
    font-weight: 500;
}

.metric-value {
    color: #f4f8ff;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-note {
    color: #4dd6b0;
    font-size: 12px;
    margin-top: 5px;
}

.winner-card {
    background: linear-gradient(120deg, #103d3b, #142f43);
    border: 1px solid #247d70;
    border-radius: 15px;
    padding: 20px;
}

.info-card {
    background: #132238;
    border: 1px solid #293f5b;
    border-radius: 13px;
    padding: 17px;
}

div[data-testid="stMetric"] {
    background: #14243a;
    border: 1px solid #29415f;
    padding: 16px;
    border-radius: 13px;
}

div[data-testid="stMetricLabel"] {
    color: #a8bdd8;
}

div[data-testid="stMetricValue"] {
    color: #f3f7ff;
}

.stButton > button {
    background: linear-gradient(90deg, #0e9f91, #16b8a3);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 9px 18px;
}

.stButton > button:hover {
    background: #087e76;
    color: white;
    border: none;
}

hr {
    border-color: #263b56;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    actual = pd.read_csv(DATA_PATH)

    lr = pd.read_csv(LR_PATH)

    rf = pd.read_csv(RF_PATH)

    actual["ordered_at"] = pd.to_datetime(
        actual["ordered_at"], errors="coerce"
    )

    lr["ordered_at"] = pd.to_datetime(
        lr["ordered_at"], errors="coerce"
    )

    rf["ordered_at"] = pd.to_datetime(
        rf["ordered_at"], errors="coerce"
    )

    for frame in [actual, lr, rf]:

        for column in [
            "actual_revenue",
            "predicted_revenue",
            "revenue_usd"
        ]:

            if column in frame.columns:
                frame[column] = pd.to_numeric(
                    frame[column], errors="coerce"
                )

    return actual, lr, rf


try:

    df, lr_predictions, rf_predictions = load_data()

except FileNotFoundError as error:

    st.error(f"Required project file not found: {error}")
    st.info(
        "Check that retail-orders.csv, predictions.csv and "
        "rf_predictions.csv exist in their respective folders."
    )
    st.stop()


# =========================================================
# MODEL METRICS - ACTUAL TRAINING OUTPUTS
# =========================================================

model_metrics = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "RMSE": [
        70.9313,
        27.9566
    ],

    "MAE": [
        39.9238,
        11.2546
    ],

    "R2 Score": [
        0.7776,
        0.9655
    ]

})


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown(
    "<h2 style='color:#eaf2ff;'>📊 Sales Intelligence</h2>",
    unsafe_allow_html=True
)

st.sidebar.caption("Retail Analytics & ML Prediction Platform")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Sales Overview",
        "Dataset & Preprocessing",
        "Model Comparison"
    ],
    index=0
)

st.sidebar.divider()

st.sidebar.markdown("### Dashboard Filters")

categories = sorted(
    df["category"].dropna().unique().tolist()
)

selected_categories = st.sidebar.multiselect(
    "Product Categories",
    options=categories,
    default=categories
)

st.sidebar.caption(
    "Choose categories to update the dashboard."
)

# Apply category filters

filtered_df = df[
    df["category"].isin(selected_categories)
].copy()

filtered_lr = lr_predictions[
    lr_predictions["category"].isin(selected_categories)
].copy()

filtered_rf = rf_predictions[
    rf_predictions["category"].isin(selected_categories)
].copy()


# =========================================================
# REUSABLE METRIC CARD
# =========================================================

def metric_card(label, value, note=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 1: SALES OVERVIEW
# =========================================================

if page == "Sales Overview":

    st.markdown(
        '<div class="main-title">Sales Revenue Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Retail business performance, '
        'revenue trends and machine learning predictions</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # ---------------- KPI CARDS ----------------

    total_revenue = filtered_df["revenue_usd"].sum()

    total_orders = filtered_df["order_id"].nunique()

    average_revenue = filtered_df["revenue_usd"].mean()

    predicted_test_revenue = filtered_rf[
        "predicted_revenue"
    ].sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "TOTAL REVENUE",
            f"${total_revenue:,.2f}",
            "Complete filtered dataset"
        )

    with c2:
        metric_card(
            "TOTAL ORDERS",
            f"{total_orders:,}",
            "Unique order IDs"
        )

    with c3:
        metric_card(
            "AVERAGE REVENUE",
            f"${average_revenue:,.2f}",
            "Average per transaction row"
        )

    with c4:
        metric_card(
            "PREDICTED TEST REVENUE",
            f"${predicted_test_revenue:,.2f}",
            "Random Forest test predictions"
        )

    st.divider()

    # ---------------- ACTUAL VS PREDICTED ----------------

    st.markdown(
        '<div class="section-title">Actual vs Predicted Revenue</div>',
        unsafe_allow_html=True
    )

    if not filtered_rf.empty:

        fig = px.scatter(
            filtered_rf,
            x="actual_revenue",
            y="predicted_revenue",
            color="category",
            hover_data=["order_id"],
            opacity=0.7,
            title="Random Forest: Actual vs Predicted Revenue"
        )

        max_value = max(
            filtered_rf["actual_revenue"].max(),
            filtered_rf["predicted_revenue"].max()
        )

        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_value,
            y1=max_value,
            line=dict(
                color="#ffffff",
                dash="dash",
                width=2
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,25,43,0.7)",
            height=440,
            xaxis_title="Actual Revenue (USD)",
            yaxis_title="Predicted Revenue (USD)"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.info("No prediction records available.")

    st.divider()

    # ---------------- MONTHLY SALES ----------------

    st.markdown(
        '<div class="section-title">Monthly Sales Trends</div>',
        unsafe_allow_html=True
    )

    if not filtered_df.empty:

        filtered_df["month"] = (
            filtered_df["ordered_at"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly = (
            filtered_df.groupby("month")["revenue_usd"]
            .sum()
            .reset_index()
        )

        fig_month = px.line(
            monthly,
            x="month",
            y="revenue_usd",
            markers=True,
            title="Monthly Revenue Performance"
        )

        fig_month.update_traces(
            line=dict(color="#24c9b0", width=3),
            marker=dict(size=7)
        )

        fig_month.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,25,43,0.7)",
            height=400,
            xaxis_title="Month",
            yaxis_title="Revenue (USD)"
        )

        st.plotly_chart(fig_month, use_container_width=True)

    st.divider()

    # ---------------- CATEGORY REVENUE ----------------

    st.markdown(
        '<div class="section-title">Revenue by Product Category</div>',
        unsafe_allow_html=True
    )

    if not filtered_df.empty:

        category_sales = (
            filtered_df.groupby("category")["revenue_usd"]
            .sum()
            .reset_index()
            .sort_values("revenue_usd", ascending=False)
        )

        fig_category = px.bar(
            category_sales,
            x="category",
            y="revenue_usd",
            color="category",
            text_auto=".2s",
            title="Category-wise Revenue Contribution"
        )

        fig_category.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,25,43,0.7)",
            height=420,
            showlegend=False,
            xaxis_title="Product Category",
            yaxis_title="Revenue (USD)"
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True
        )

    st.divider()

    st.markdown(
        """
        <div style="text-align:center;color:#8297b3;font-size:12px;">
            Sales Revenue Prediction | Apache Spark • PySpark •
            Random Forest • Streamlit • Plotly
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE 2: DATASET & PREPROCESSING
# =========================================================

elif page == "Dataset & Preprocessing":

    st.markdown(
        '<div class="main-title">Dataset & Preprocessing</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Dataset exploration, data quality, '
        'cleaning and feature engineering</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # ---------------- DATASET OVERVIEW ----------------

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    rows, columns = df.shape

    total_missing = int(df.isnull().sum().sum())

    duplicate_count = int(df.duplicated().sum())

    unique_orders = df["order_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("TOTAL RECORDS", f"{rows:,}", "Retail transaction rows")

    with c2:
        metric_card("TOTAL COLUMNS", str(columns), "Dataset features")

    with c3:
        metric_card("MISSING VALUES", str(total_missing), "Across all columns")

    with c4:
        metric_card("UNIQUE ORDERS", f"{unique_orders:,}", "Distinct order IDs")

    st.divider()

    # ---------------- DATASET DESCRIPTION ----------------

    st.markdown(
        '<div class="section-title">Dataset Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <b>Dataset:</b> Retail Orders Dataset<br><br>
        <b>Target Variable:</b> revenue_usd<br><br>
        <b>Problem Type:</b> Regression<br><br>
        <b>Technology:</b> Apache Spark and PySpark<br><br>
        <b>Prediction Objective:</b> Estimate transaction-level sales revenue.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ---------------- COLUMN INFORMATION ----------------

    st.markdown(
        '<div class="section-title">Column Names and Data Types</div>',
        unsafe_allow_html=True
    )

    schema_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": [
            int(df[col].isnull().sum()) for col in df.columns
        ],
        "Unique Values": [
            int(df[col].nunique()) for col in df.columns
        ]
    })

    st.dataframe(
        schema_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    

    
    # ---------------- DATA DICTIONARY ----------------

    st.markdown(
        '<div class="section-title">Data Dictionary - Understanding Every Column</div>',
        unsafe_allow_html=True
    )

    data_dictionary = pd.DataFrame([
        {
            "Column": "order_id",
            "Description": "Unique identifier assigned to an order.",
            "Example": "O500860",
            "Purpose": "Identifies the order."
        },
        {
            "Column": "line_no",
            "Description": "Sequence number of a product line within an order; 1 is the first line, 2 the second, and so on. It is not a product ID.",
            "Example": "1, 2, 3",
            "Purpose": "Distinguishes multiple product lines in one order."
        },
        {
            "Column": "ordered_at",
            "Description": "Date and time when the order was placed.",
            "Example": "2023-01-01 08:28",
            "Purpose": "Used to extract month, weekday and hour."
        },
        {
            "Column": "customer_id",
            "Description": "Unique identifier of the customer.",
            "Example": "C5122",
            "Purpose": "Identifies the customer."
        },
        {
            "Column": "channel",
            "Description": "Platform through which the order was placed.",
            "Example": "android, web, ios",
            "Purpose": "Categorical business feature."
        },
        {
            "Column": "country",
            "Description": "Country code associated with the transaction.",
            "Example": "JP, US, BR",
            "Purpose": "Geographical categorical feature."
        },
        {
            "Column": "category",
            "Description": "Product category associated with the order line.",
            "Example": "beauty, grocery",
            "Purpose": "Categorical product feature."
        },
        {
            "Column": "units",
            "Description": "Number of product units purchased.",
            "Example": "4",
            "Purpose": "Numerical model feature."
        },
        {
            "Column": "unit_price_usd",
            "Description": "Price of one product unit in US dollars.",
            "Example": "$26.09",
            "Purpose": "Numerical model feature."
        },
        {
            "Column": "discount_pct",
            "Description": "Discount applied to the product, stored as a decimal.",
            "Example": "0.20 = 20%",
            "Purpose": "Numerical model feature."
        },
        {
            "Column": "revenue_usd",
            "Description": "Revenue generated after applying the discount.",
            "Example": "$83.49",
            "Purpose": "Target variable for regression."
        },
        {
            "Column": "returned",
            "Description": "Indicates whether the purchased product was returned.",
            "Example": "True / False",
            "Purpose": "Categorical model feature."
        }
    ])

    st.dataframe(
        data_dictionary,
        use_container_width=True,
        hide_index=True,
        height=520
    )

    st.download_button(
        "Download Data Dictionary",
        data=data_dictionary.to_csv(index=False).encode("utf-8"),
        file_name="retail_dataset_data_dictionary.csv",
        mime="text/csv"
    )

    st.divider()


    # ---------------- MISSING VALUES ----------------

    st.markdown(
        '<div class="section-title">Missing Values Analysis</div>',
        unsafe_allow_html=True
    )

    missing_df = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing_df.columns = ["Column", "Missing Count"]

    missing_df = missing_df.sort_values(
        "Missing Count", ascending=False
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            "Total Missing Values",
            f"{total_missing:,}"
        )

        st.metric(
            "Duplicate Rows in Loaded Dataset",
            f"{duplicate_count:,}"
        )

    with col2:

        fig_missing = px.bar(
            missing_df,
            x="Column",
            y="Missing Count",
            title="Missing Values by Column",
            color="Missing Count",
            color_continuous_scale="Teal"
        )

        fig_missing.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,25,43,0.7)",
            height=350
        )

        st.plotly_chart(
            fig_missing,
            use_container_width=True
        )

    if total_missing == 0:
        st.success(
            "No missing values found in the original dataset."
        )

    st.divider()

    # ---------------- NUMERIC SUMMARY ----------------

    st.markdown(
        '<div class="section-title">Numerical Features Summary</div>',
        unsafe_allow_html=True
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    st.dataframe(
        numeric_df.describe().T.round(2),
        use_container_width=True
    )

    st.divider()

    # ---------------- CATEGORICAL DISTRIBUTION ----------------

    st.markdown(
        '<div class="section-title">Categorical Feature Distribution</div>',
        unsafe_allow_html=True
    )

    categorical_columns = [
        "category", "country", "channel", "returned"
    ]

    selected_column = st.selectbox(
        "Select categorical feature",
        categorical_columns
    )

    distribution = (
        df[selected_column]
        .fillna("Unknown")
        .value_counts()
        .reset_index()
    )

    distribution.columns = ["Value", "Count"]

    fig_distribution = px.bar(
        distribution,
        x="Value",
        y="Count",
        color="Value",
        text="Count",
        title=f"Distribution of {selected_column}"
    )

    fig_distribution.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,25,43,0.7)",
        height=400,
        showlegend=False
    )

    st.plotly_chart(
        fig_distribution,
        use_container_width=True
    )

    st.divider()

    # ---------------- PREPROCESSING WORKFLOW ----------------

    st.markdown(
        '<div class="section-title">Data Preprocessing Workflow</div>',
        unsafe_allow_html=True
    )

    preprocessing_steps = pd.DataFrame({
        "Step": [
            "1. Data Loading",
            "2. Data Inspection",
            "3. Duplicate Removal",
            "4. Missing Value Handling",
            "5. Date Feature Extraction",
            "6. Categorical Encoding",
            "7. Feature Assembly",
            "8. Train-Test Split"
        ],

        "Description": [
            "Loaded retail-orders.csv using PySpark.",
            "Inspected dataset dimensions, schema and data types.",
            "Removed duplicate records using dropDuplicates().",
            "Removed rows with missing target and handled missing feature values.",
            "Extracted order month, day of week and hour from ordered_at.",
            "Applied StringIndexer and OneHotEncoder to categorical columns.",
            "Combined numerical and encoded categorical features using VectorAssembler.",
            "Split dataset into 80% training and 20% testing using seed 42."
        ],

        "Status": [
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Completed",
            "Applied in model pipeline",
            "Completed",
            "Completed"
        ]
    })

    st.dataframe(
        preprocessing_steps,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ---------------- FEATURE DETAILS ----------------

    st.markdown(
        '<div class="section-title">Model Features</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Numerical Features")

        st.write([
            "units",
            "unit_price_usd",
            "discount_pct",
            "order_month",
            "order_day",
            "order_hour"
        ])

    with col2:

        st.markdown("#### Categorical Features")

        st.write([
            "category",
            "country",
            "channel",
            "returned"
        ])

    st.info(
        "Target variable: revenue_usd. "
        "The country field is used as the geographical categorical "
        "feature because the dataset contains country, not region."
    )

    st.divider()

    # ---------------- SAMPLE DATA ----------------

    st.markdown(
        '<div class="section-title">Dataset Sample Records</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Dataset Summary",
        data=schema_df.to_csv(index=False).encode("utf-8"),
        file_name="dataset_summary.csv",
        mime="text/csv"
    )

    st.caption(
        "Dataset exploration and preprocessing | "
        "Apache Spark and PySpark"
    )


# =========================================================
# PAGE 3: MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.markdown(
        '<div class="main-title">Model Performance Comparison</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Compare Linear Regression and '
        'Random Forest on the same held-out test split</div>',
        unsafe_allow_html=True
    )

    st.write("")

    # ---------------- WINNER ----------------

    st.markdown(
        """
        <div class="winner-card">
            <h3 style="color:#5de2c5;margin:0;">
                🏆 Best Performing Model: Random Forest Regressor
            </h3>
            <p style="color:#c3d9e9;margin-bottom:0;">
                It achieved the lowest RMSE and MAE and the highest
                R² score among the two tested models.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # ---------------- METRIC CARDS ----------------

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "RANDOM FOREST R²",
            "96.55%",
            "Higher explanatory performance"
        )

    with c2:
        metric_card(
            "RANDOM FOREST MAE",
            "$11.25",
            "Lower average absolute error"
        )

    with c3:
        metric_card(
            "RANDOM FOREST RMSE",
            "$27.96",
            "Lower root mean squared error"
        )

    st.divider()

    # ---------------- COMPARISON TABLE ----------------

    st.markdown(
        '<div class="section-title">Model Metrics Comparison</div>',
        unsafe_allow_html=True
    )

    display_metrics = model_metrics.copy()

    display_metrics["RMSE"] = display_metrics["RMSE"].map(
        lambda x: f"${x:.4f}"
    )

    display_metrics["MAE"] = display_metrics["MAE"].map(
        lambda x: f"${x:.4f}"
    )

    display_metrics["R2 Score"] = display_metrics["R2 Score"].map(
        lambda x: f"{x * 100:.2f}%"
    )

    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Metrics are from the original Spark evaluation outputs. "
        "RMSE and MAE are in USD."
    )

    st.divider()

    # ---------------- METRIC COMPARISON CHART ----------------

    st.markdown(
        '<div class="section-title">Error Metric Comparison</div>',
        unsafe_allow_html=True
    )

    error_data = model_metrics.melt(
        id_vars="Model",
        value_vars=["RMSE", "MAE"],
        var_name="Metric",
        value_name="Error (USD)"
    )

    fig_error = px.bar(
        error_data,
        x="Metric",
        y="Error (USD)",
        color="Model",
        barmode="group",
        text_auto=".2f",
        title="RMSE and MAE Comparison (Lower is Better)",
        color_discrete_sequence=["#f2a65a", "#20c9a8"]
    )

    fig_error.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,25,43,0.7)",
        height=420
    )

    st.plotly_chart(
        fig_error,
        use_container_width=True
    )

    # ---------------- R2 COMPARISON ----------------

    st.markdown(
        '<div class="section-title">R² Score Comparison</div>',
        unsafe_allow_html=True
    )

    fig_r2 = px.bar(
        model_metrics,
        x="Model",
        y="R2 Score",
        color="Model",
        text=model_metrics["R2 Score"].map(
            lambda x: f"{x * 100:.2f}%"
        ),
        title="Coefficient of Determination (Higher is Better)",
        color_discrete_sequence=["#f2a65a", "#20c9a8"]
    )

    fig_r2.update_yaxes(
        range=[0, 1],
        tickformat=".0%"
    )

    fig_r2.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,25,43,0.7)",
        height=380,
        showlegend=False
    )

    st.plotly_chart(
        fig_r2,
        use_container_width=True
    )

    st.divider()

    # ---------------- MODEL PREDICTION SCATTER ----------------

    st.markdown(
        '<div class="section-title">Prediction Comparison</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### Linear Regression")

        if not filtered_lr.empty:

            fig_lr = px.scatter(
                filtered_lr,
                x="actual_revenue",
                y="predicted_revenue",
                color="category",
                opacity=0.65,
                hover_data=["order_id"],
                title="Linear Regression Predictions"
            )

            max_lr = max(
                filtered_lr["actual_revenue"].max(),
                filtered_lr["predicted_revenue"].max()
            )

            fig_lr.add_shape(
                type="line",
                x0=0, y0=0,
                x1=max_lr, y1=max_lr,
                line=dict(color="white", dash="dash")
            )

            fig_lr.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(12,25,43,0.7)",
                height=400
            )

            st.plotly_chart(fig_lr, use_container_width=True)

        else:

            st.warning("Linear Regression predictions unavailable.")

    with right:

        st.markdown("#### Random Forest")

        if not filtered_rf.empty:

            fig_rf = px.scatter(
                filtered_rf,
                x="actual_revenue",
                y="predicted_revenue",
                color="category",
                opacity=0.65,
                hover_data=["order_id"],
                title="Random Forest Predictions"
            )

            max_rf = max(
                filtered_rf["actual_revenue"].max(),
                filtered_rf["predicted_revenue"].max()
            )

            fig_rf.add_shape(
                type="line",
                x0=0, y0=0,
                x1=max_rf, y1=max_rf,
                line=dict(color="white", dash="dash")
            )

            fig_rf.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(12,25,43,0.7)",
                height=400
            )

            st.plotly_chart(fig_rf, use_container_width=True)

        else:

            st.warning("Random Forest predictions unavailable.")

    st.divider()

    # ---------------- DOWNLOAD COMPARISON ----------------

    st.markdown(
        '<div class="section-title">Download Model Comparison</div>',
        unsafe_allow_html=True
    )

    csv_data = model_metrics.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Model Metrics CSV",
        data=csv_data,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

    st.info(
        "Random Forest is selected as the final model because "
        "it performed better on all three reported evaluation metrics."
    )

    st.caption(
        "Note: This project predicts transaction-level revenue. "
        "It should not be interpreted as a fully validated future-sales "
        "forecasting system."
    )

    st.divider()

    st.markdown(
        """
        <div style="text-align:center;color:#8297b3;font-size:12px;">
            Model Comparison | Linear Regression vs Random Forest
        </div>
        """,
        unsafe_allow_html=True
    )
