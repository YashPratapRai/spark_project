
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# 1. Create Spark Session
spark = (
    SparkSession.builder
    .appName("SalesRevenuePreprocessing")
    .master("local[*]")
    .getOrCreate()
)

# 2. Load Dataset
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/retail-orders.csv")
)

df = df.dropDuplicates()

print("Original Dataset:")
df.show(5, truncate=False)

# 3. Convert Boolean column into String
df = df.withColumn("returned", col("returned").cast("string"))

# 4. Define categorical columns
categorical_columns = [
    "category",
    "country",
    "channel",
    "returned"
]

# 5. StringIndexer
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in categorical_columns
]

# 6. OneHotEncoder
indexed_columns = [c + "_index" for c in categorical_columns]

encoded_columns = [c + "_encoded" for c in categorical_columns]

encoder = OneHotEncoder(
    inputCols=indexed_columns,
    outputCols=encoded_columns,
    handleInvalid="keep"
)

# 7. Create Pipeline
pipeline = Pipeline(stages=indexers + [encoder])

# 8. Fit and Transform Dataset
model = pipeline.fit(df)
encoded_df = model.transform(df)

# 9. Display Encoded Features
print("\nEncoded Dataset:")

encoded_df.select(
    "category",
    "category_index",
    "category_encoded",
    "country",
    "country_encoded",
    "channel",
    "channel_encoded",
    "returned",
    "returned_encoded"
).show(10, truncate=False)

# 10. Save Encoded Dataset
encoded_df.write.mode("overwrite").parquet(
    "outputs/encoded_data"
)

print("\nTask 2 Completed Successfully!")

spark.stop()
