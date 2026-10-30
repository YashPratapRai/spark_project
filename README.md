# 📊 Sales Revenue Prediction Using Apache Spark (PySpark)

A retail analytics and machine learning project that uses **Apache Spark (PySpark)** to process retail transaction data, train regression models, predict sales revenue, and visualize business insights through an interactive **Streamlit dashboard**.

## 🚀 Live Dashboard

**Live App:** Add your Streamlit deployment URL here after deployment.

**GitHub Repository:** https://github.com/YashPratapRai/spark_project

---

## 📌 Project Overview

Retail businesses generate large volumes of transaction data containing information about product categories, quantities, prices, discounts, sales channels, countries, and order timestamps.

This project uses Apache Spark and its MLlib library to analyze retail transaction data and estimate revenue at the transaction level.

Two machine learning algorithms are implemented and evaluated:

- Linear Regression
- Random Forest Regression

The models are compared using RMSE, MAE, and R² Score. A Streamlit dashboard presents sales insights, dataset information, and model performance.

## 🎯 Objectives

- Load and process retail transaction data using Apache Spark.
- Perform data cleaning and preprocessing.
- Handle categorical variables using StringIndexer and OneHotEncoder.
- Perform feature engineering using order timestamp information.
- Train regression models using Spark MLlib.
- Evaluate and compare model performance.
- Analyze prediction errors across product categories.
- Build an interactive dashboard using Streamlit and Plotly.

## 📂 Dataset Description

| Attribute | Details |
|---|---|
| Dataset | Retail Orders |
| Source | Datanemics |
| Total Records | 7,000 |
| Total Columns | 12 |
| Data Type | Synthetic retail transaction data |
| Period | January 2023 – December 2024 |
| Target Variable | `revenue_usd` |
| Currency | USD |

**Dataset Source:** https://datanemics.com/datasets/retail-orders/

### Important Features

- `order_id` – Unique order identifier
- `ordered_at` – Order timestamp
- `channel` – Sales channel
- `country` – Country associated with the transaction
- `category` – Product category
- `units` – Quantity sold
- `unit_price_usd` – Unit price
- `discount_pct` – Discount percentage
- `returned` – Return status
- `revenue_usd` – Target variable

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Apache Spark | Distributed data processing |
| PySpark | Python API for Spark |
| Spark MLlib | Machine learning |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Plotly | Interactive charts |
| Streamlit | Interactive web dashboard |
| Git & GitHub | Version control |

## ⚙️ Project Workflow

1. **Data Loading:** Load the retail orders CSV file into a Spark DataFrame.
2. **Data Exploration:** Inspect dataset dimensions, schema, data types, and missing values.
3. **Data Cleaning:** Remove duplicate records and rows with missing target values.
4. **Preprocessing:** Encode categorical features using StringIndexer and OneHotEncoder.
5. **Feature Engineering:** Extract month, day, and hour from the order timestamp.
6. **Model Training:** Train Linear Regression and Random Forest Regression models.
7. **Model Evaluation:** Evaluate models using RMSE, MAE, and R² Score.
8. **Error Analysis:** Analyze prediction errors across categories and revenue ranges.
9. **Dashboard Development:** Visualize sales trends, category performance, and model comparisons using Streamlit.

## 🤖 Model Performance

Both models were trained and evaluated using the same train-test split.

- Training records: 5,663
- Testing records: 1,337
- Random seed: 42

### Model Comparison

| Evaluation Metric | Linear Regression | Random Forest |
|---|---:|---:|
| RMSE | 70.9313 | **27.9566** |
| MAE | 39.9238 | **11.2546** |
| R² Score | 0.7776 | **0.9655** |

### 🏆 Best Performing Model: Random Forest

Random Forest achieved better performance across all three evaluation metrics.

- Lower RMSE indicates smaller prediction errors.
- Lower MAE indicates better average prediction accuracy.
- Higher R² indicates that the model explains a larger proportion of the variation in revenue.

The Random Forest model was therefore selected as the final model.

## 📊 Interactive Dashboard

The Streamlit dashboard provides three main sections:

### 1. Sales Overview
- Total revenue
- Total orders
- Average transaction revenue
- Predicted test revenue
- Monthly revenue trends
- Category-wise revenue analysis
- Actual vs. predicted revenue visualization

### 2. Dataset & Preprocessing
- Dataset overview
- Data quality analysis
- Dataset information
- Feature and preprocessing details

### 3. Model Comparison
- Linear Regression performance
- Random Forest performance
- RMSE comparison
- MAE comparison
- R² Score comparison
- Best model identification

## 📁 Repository Structure

```text
spark_project/
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── retail-orders.csv
│
├── outputs/
│   ├── predictions.csv
│   ├── rf_predictions.csv
│   ├── metrics.txt
│   └── rf_metrics.txt
│
├── src/
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── compare_models.py
│   └── analyze_errors.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 💻 Installation & Execution

### Prerequisites

- Python 3.11
- Java 21
- Apache Spark 4.2.0
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YashPratapRai/spark_project.git
cd spark_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For running the Spark training scripts, install PySpark as well:

```bash
pip install pyspark==4.2.0
```

### 4. Configure Apache Spark

Set the Spark environment variables in PowerShell:

```powershell
$env:SPARK_HOME = "C:\spark\spark-4.2.0-bin-hadoop3"
$env:PATH = "$env:SPARK_HOME\bin;$env:PATH"
```

Make sure Java and Spark are correctly installed and configured.

### 5. Run the data processing and model scripts

```bash
python src/load_data.py
python src/preprocessing.py
python src/train_model.py
python src/compare_models.py
python src/analyze_errors.py
```

### 6. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser, usually at:

`http://localhost:8501`

## 📈 Key Findings

- The dataset contains 7,000 retail transaction records.
- Random Forest outperformed Linear Regression on the held-out test dataset.
- Random Forest achieved an R² Score of 0.9655.
- Electronics transactions showed comparatively higher prediction errors.
- Prediction errors increased for transactions with higher actual revenue.

## ⚠️ Limitations & Future Improvements

- The dataset is synthetic and may not represent actual retail business behavior.
- The current implementation predicts revenue for individual held-out transaction records; it is not a time-series forecast of future monthly or yearly sales.
- The `returned` feature may introduce information leakage if used to predict revenue before an order is completed.
- Future work can explore time-series forecasting, hyperparameter tuning, feature selection, and evaluation on real-world retail datasets.
- Model performance should be validated on independent real-world data before business use.

## 🔮 Future Scope

- Implement time-series forecasting for future sales revenue.
- Integrate real-time retail transaction data.
- Experiment with Gradient Boosting and other regression algorithms.
- Optimize model hyperparameters.
- Deploy the application with automated model retraining.
- Extend the dashboard with additional business intelligence features.

## 👨‍💻 Author

**Yash Pratap Rai**

B.Tech – Computer Science and Engineering  
Lovely Professional University

**Course:** INT315 – Cluster Computing

**Project:** Sales Revenue Prediction Using Apache Spark (PySpark)

---

⭐ If you find this project interesting, consider giving the repository a star!
