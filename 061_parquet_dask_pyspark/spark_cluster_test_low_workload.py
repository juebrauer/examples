from pyspark.sql import SparkSession
from pyspark.sql import functions as F


spark = (
    SparkSession.builder
    .appName("MiniClusterTest")
    .getOrCreate()
)

print("Spark version:", spark.version)
print("spark=", spark)

df = spark.range(0, 50_000_000)

result = (
    df
    .withColumn("group", F.col("id") % 10)
    .groupBy("group")
    .count()
    .orderBy("group")
)

result.show()

spark.stop()