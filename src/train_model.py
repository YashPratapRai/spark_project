
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, month, dayofweek, hour, count, when
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import os


# =====================================================
# 1. CREATE SPARK SESSION
# =====================================================

spark = (
    SparkSession.builder
    .appName("SalesRevenuePrediction")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\nSpark Session Created Successfully!")


# =====================================================
# 2. LOAD DATASET
# =====================================================

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/retail-orders.csv")
)

print("\n========== DATASET INFORMATION ==========")

print("Total Records:", df.count())
print("Total Columns:", len(df.columns))

df.printSchema()


# =====================================================
# 3. DATA CLEANING
# =====================================================

df = df.dropDuplicates()

# Remove records with missing target
df = df.filter(col("revenue_usd").isNotNull())

# Remove invalid numeric values in required features
required_numeric = [
    "units",
    "unit_price_usd",
    "discount_pct"
]

for c in required_numeric:
    df = df.filter(col(c).isNotNull())

print("\nRecords After Cleaning:", df.count())


# =====================================================
# 4. FEATURE ENGINEERING
# =====================================================

df = (
    df.withColumn("order_month", month("ordered_at"))
      .withColumn("order_day", dayofweek("ordered_at"))
      .withColumn("order_hour", hour("ordered_at"))
      .withColumn("returned", col("returned").cast("string"))
)

print("\nFeature Engineering Completed!")


# =====================================================
# 5. DEFINE FEATURES
# =====================================================

categorical_cols = [
    "category",
    "country",
    "channel",
    "returned"
]

numeric_cols = [
    "units",
    "unit_price_usd",
    "discount_pct",
    "order_month",
    "order_day",
    "order_hour"
]


# =====================================================
# 6. HANDLE MISSING FEATURE VALUES
# =====================================================

for c in numeric_cols:

    mean_value = df.selectExpr(
        f"avg(`{c}`)"
    ).first()[0]

    if mean_value is not None:
        df = df.na.fill({c: mean_value})


for c in categorical_cols:
    df = df.na.fill({c: "Unknown"})


# =====================================================
# 7. TRAIN TEST SPLIT
# =====================================================

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\n========== TRAIN TEST SPLIT ==========")

print("Training Records:", train_df.count())
print("Testing Records:", test_df.count())


# =====================================================
# 8. STRING INDEXER
# =====================================================

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in categorical_cols
]


# =====================================================
# 9. ONE HOT ENCODER
# =====================================================

encoder = OneHotEncoder(
    inputCols=[
        c + "_index" for c in categorical_cols
    ],
    outputCols=[
        c + "_encoded" for c in categorical_cols
    ],
    handleInvalid="keep"
)


# =====================================================
# 10. VECTOR ASSEMBLER
# =====================================================

assembler = VectorAssembler(
    inputCols=(
        numeric_cols +
        [c + "_encoded" for c in categorical_cols]
    ),
    outputCol="features",
    handleInvalid="keep"
)


# =====================================================
# 11. LINEAR REGRESSION MODEL
# =====================================================

lr = LinearRegression(
    featuresCol="features",
    labelCol="revenue_usd",
    predictionCol="predicted_revenue",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)


# =====================================================
# 12. CREATE ML PIPELINE
# =====================================================

pipeline = Pipeline(
    stages=indexers + [
        encoder,
        assembler,
        lr
    ]
)


# =====================================================
# 13. TRAIN MODEL
# =====================================================

print("\n========== MODEL TRAINING ==========")

print("Training Linear Regression Model...")

model = pipeline.fit(train_df)

print("Model Training Completed Successfully!")


# =====================================================
# 14. GENERATE TEST PREDICTIONS
# =====================================================

print("\nGenerating Test Predictions...")

predictions = model.transform(test_df)


# =====================================================
# 15. MODEL EVALUATION
# =====================================================

evaluator = RegressionEvaluator(
    labelCol="revenue_usd",
    predictionCol="predicted_revenue"
)

rmse = evaluator.evaluate(
    predictions,
    {evaluator.metricName: "rmse"}
)

mae = evaluator.evaluate(
    predictions,
    {evaluator.metricName: "mae"}
)

r2 = evaluator.evaluate(
    predictions,
    {evaluator.metricName: "r2"}
)


print("\n========== MODEL EVALUATION RESULTS ==========")

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")


# =====================================================
# 16. DISPLAY ACTUAL VS PREDICTED REVENUE
# =====================================================

result_df = predictions.select(
    "order_id",
    "ordered_at",
    "category",
    col("revenue_usd").alias("actual_revenue"),
    col("predicted_revenue")
)

print("\n========== SAMPLE PREDICTIONS ==========")

result_df.show(10, truncate=False)


# =====================================================
# 17. EXPORT PREDICTIONS TO CSV
# =====================================================

os.makedirs("outputs", exist_ok=True)

pandas_df = result_df.toPandas()

pandas_df["predicted_revenue"] = (
    pandas_df["predicted_revenue"].clip(lower=0)
)

pandas_df.to_csv(
    "outputs/predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("File: outputs/predictions.csv")


# =====================================================
# 18. SAVE MODEL METRICS
# =====================================================

with open("outputs/metrics.txt", "w") as file:

    file.write("SALES REVENUE PREDICTION\n")
    file.write("========================\n\n")

    file.write("Algorithm: Linear Regression\n")
    file.write("Training Records: " + str(train_df.count()) + "\n")
    file.write("Testing Records: " + str(test_df.count()) + "\n")

    file.write(f"RMSE: {rmse:.4f}\n")
    file.write(f"MAE: {mae:.4f}\n")
    file.write(f"R2 Score: {r2:.4f}\n")


print("\nMetrics saved successfully!")
print("File: outputs/metrics.txt")


# =====================================================
# 19. FINAL STATUS
# =====================================================

print("\n==========================================")
print("TASK 3: MODEL TRAINING COMPLETED")
print("TASK 4: MODEL EVALUATION COMPLETED")
print("TASK 5: PREDICTIONS GENERATED")
print("==========================================")


# =====================================================
# 20. STOP SPARK SESSION
# =====================================================

spark.stop()

print("\nSpark Session Stopped Successfully!")
