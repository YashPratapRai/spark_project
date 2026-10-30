
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan
from pyspark.sql.types import NumericType

# 1. Create Spark Session
spark = (
    SparkSession.builder
    .appName("SalesRevenuePrediction")
    .master("local[*]")
    .getOrCreate()
)

print("Spark Session Created Successfully!")

# 2. Load Dataset
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/retail-orders.csv")
)

# 3. Dataset Overview
print("\n========== DATASET OVERVIEW ==========")

print("Total Rows:", df.count())
print("Total Columns:", len(df.columns))

print("\nDataset Schema:")
df.printSchema()

print("\nFirst 5 Records:")
df.show(5, truncate=False)

# 4. Missing Value Analysis
print("\n========== MISSING VALUE ANALYSIS ==========")

null_expressions = []

for field in df.schema.fields:
    c = field.name

    if isinstance(field.dataType, NumericType):
        null_expressions.append(
            count(when(col(c).isNull() | isnan(col(c)), 1)).alias(c)
        )
    else:
        null_expressions.append(
            count(when(col(c).isNull(), 1)).alias(c)
        )

df.select(null_expressions).show(truncate=False)

# 5. Remove rows where target is missing
if "revenue_usd" in df.columns:
    df = df.filter(col("revenue_usd").isNotNull())

# 6. Handle missing values in input features
numeric_columns = [
    field.name
    for field in df.schema.fields
    if isinstance(field.dataType, NumericType)
    and field.name not in [
        "order_id", "line_no", "customer_id", "revenue_usd"
    ]
]

categorical_columns = [
    field.name
    for field in df.schema.fields
    if field.dataType.simpleString() == "string"
]

# Fill numerical missing values with mean
for c in numeric_columns:
    mean_value = df.select(col(c)).na.drop().agg({c: "avg"}).first()[0]

    if mean_value is not None:
        df = df.na.fill({c: mean_value})

# Fill categorical missing values
for c in categorical_columns:
    df = df.na.fill({c: "Unknown"})

# 7. Remove Duplicate Records
df = df.dropDuplicates()

# 8. Final Dataset
print("\n========== CLEANED DATASET ==========")

print("Final Rows:", df.count())
print("Final Columns:", len(df.columns))

df.show(5, truncate=False)

print("\nTask 1 Completed Successfully!")

# 9. Stop Spark
spark.stop()
