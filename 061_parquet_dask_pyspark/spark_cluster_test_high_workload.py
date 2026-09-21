from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("MiniClusterTest")
    .getOrCreate()
)

# Genug Daten, damit Spark etwas zu tun hat
df = spark.range(0, 5_000_000_000)

# Erste Aggregation
a = (
    df
    .withColumn("group1", F.col("id") % 1000)
    .withColumn("value", (F.col("id") * 17) % 10000)
    .groupBy("group1")
    .agg(
        F.sum("value").alias("sum_value"),
        F.avg("value").alias("avg_value"),
        F.count("*").alias("count_value")
    )
)

# Zweite Aggregation
b = (
    df
    .withColumn("group1", F.col("id") % 1000)
    .withColumn("group2", F.col("id") % 100)
    .groupBy("group1", "group2")
    .agg(
        F.max("id").alias("max_id"),
        F.min("id").alias("min_id")
    )
)

# Join + weitere Aggregation + Sortierung
result = (
    b
    .join(a, on="group1")
    .groupBy("group2")
    .agg(
        F.sum("sum_value").alias("total_sum"),
        F.avg("avg_value").alias("mean_avg"),
        F.sum("count_value").alias("total_count"),
        F.max("max_id").alias("max_id")
    )
    .orderBy(F.desc("total_sum"))
)

result.show(100)

spark.stop()