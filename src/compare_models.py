
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, month, dayofweek, hour,
    abs as spark_abs, avg, max as spark_max
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("RandomForestRevenuePrediction") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# Create required directories
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# 2. Load and clean data
df = spark.read.csv(
    "data/retail-orders.csv",
    header=True,
    inferSchema=True
)

df = df.dropDuplicates()
df = df.filter(col("revenue_usd").isNotNull())

df = df.filter(
    col("units").isNotNull() &
    col("unit_price_usd").isNotNull() &
    col("discount_pct").isNotNull()
)


# 3. Feature engineering
df = df.withColumn("order_month", month("ordered_at"))
df = df.withColumn("order_day", dayofweek("ordered_at"))
df = df.withColumn("order_hour", hour("ordered_at"))

df = df.withColumn("returned", col("returned").cast("string"))


# 4. Handle missing values
categorical_cols = ["category", "country", "channel", "returned"]

numeric_cols = [
    "units",
    "unit_price_usd",
    "discount_pct",
    "order_month",
    "order_day",
    "order_hour"
]

for c in categorical_cols:
    df = df.na.fill({c: "Unknown"})

for c in numeric_cols:
    mean_value = df.selectExpr(
        f"avg({c}) as mean_value"
    ).first()["mean_value"]

    df = df.na.fill({c: mean_value})


# 5. Train-test split
train_data, test_data = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

train_count = train_data.count()
test_count = test_data.count()

print("\n========== RANDOM FOREST TRAINING ==========")
print("Training Records:", train_count)
print("Testing Records:", test_count)


# 6. Encode categorical features
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoder = OneHotEncoder(
    inputCols=[c + "_index" for c in categorical_cols],
    outputCols=[c + "_encoded" for c in categorical_cols],
    handleInvalid="keep"
)


# 7. Assemble features
assembler = VectorAssembler(
    inputCols=numeric_cols + [
        c + "_encoded" for c in categorical_cols
    ],
    outputCol="features"
)


# 8. Random Forest model
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="revenue_usd",
    numTrees=150,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(
    stages=indexers + [encoder, assembler, rf]
)


# 9. Train model
model = pipeline.fit(train_data)

print("\nRandom Forest Training Completed!")


# 10. SAVE TRAINED RANDOM FOREST MODEL
rf_model_path = os.path.abspath(
    "models/random_forest_model"
)

model.write().overwrite().save(rf_model_path)

print("\nRandom Forest model saved successfully!")
print("Model location:", rf_model_path)


# 11. Generate predictions
predictions = model.transform(test_data)


# 12. Model evaluation
evaluator = RegressionEvaluator(
    labelCol="revenue_usd",
    predictionCol="prediction"
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

print("\n========== RANDOM FOREST RESULTS ==========")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")


# 13. Category-wise error analysis
category_errors = predictions.withColumn(
    "absolute_error",
    spark_abs(col("revenue_usd") - col("prediction"))
)

print("\n========== CATEGORY-WISE ERROR ==========")

category_errors.groupBy("category").agg(
    avg("absolute_error").alias("Average_Error"),
    spark_max("absolute_error").alias("Maximum_Error")
).orderBy(
    col("Average_Error").desc()
).show()


# 14. Save predictions
result = predictions.select(
    "order_id",
    "ordered_at",
    "category",
    col("revenue_usd").alias("actual_revenue"),
    col("prediction").alias("predicted_revenue")
)

pdf = result.toPandas()

# Keep exported predictions non-negative
pdf["predicted_revenue"] = pdf["predicted_revenue"].clip(lower=0)

pdf.to_csv(
    "outputs/rf_predictions.csv",
    index=False
)


# 15. Save evaluation metrics
with open("outputs/rf_metrics.txt", "w") as f:
    f.write("Model: Random Forest Regressor\n")
    f.write(f"Training Records: {train_count}\n")
    f.write(f"Testing Records: {test_count}\n")
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")


print("\n========== FILES SAVED ==========")
print("Random Forest model:", rf_model_path)
print("Predictions: outputs/rf_predictions.csv")
print("Metrics: outputs/rf_metrics.txt")


# 16. Stop Spark
spark.stop()

print("\nProject execution completed successfully!")
