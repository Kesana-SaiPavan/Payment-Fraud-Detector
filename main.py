# Payment Fraud Detector - Real-Time Simulation
# Built for Data Engineer interviews (Fintech focus)
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import shutil
import os

# Clean old data (for demo)
if os.path.exists("delta_lake"): shutil.rmtree("delta_lake")
if os.path.exists("output"): shutil.rmtree("output")

spark = SparkSession.builder \
    .appName("RealTimeFraudDetector") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Schema for payment transactions
schema = StructType([
    StructField("transaction_id", StringType()),
    StructField("user_id", StringType()),
    StructField("amount", DoubleType()),
    StructField("timestamp", TimestampType()),
    StructField("merchant", StringType()),
    StructField("country", StringType()),
    StructField("device", StringType())
])

# Read sample data
df = spark.read.csv("data/sample_payments.csv", header=True, schema=schema)

# Fraud Rules (exactly what banks use)
flagged = df \
    .withColumn("is_high_amount", when(col("amount") > 5000, 1).otherwise(0)) \
    .withColumn("is_new_device", when(col("device").contains("new"), 1).otherwise(0)) \
    .withColumn("is_velocity_risk", when(col("amount") > 3000, 1).otherwise(0)) \
    .withColumn("is_international", when(col("country") != "USA", 1).otherwise(0)) \
    .withColumn("fraud_score", 
                col("is_high_amount")*40 + 
                col("is_new_device")*30 + 
                col("is_velocity_risk")*20 + 
                col("is_international")*10) \
    .withColumn("is_fraud", when(col("fraud_score") >= 60, "YES").otherwise("NO"))

# Gold layer: clean + fraud alerts
clean = flagged.filter(col("is_fraud") == "NO").drop("fraud_score", "is_*")
alerts = flagged.filter(col("is_fraud") == "YES")

# Save as Delta Lake (industry standard)
clean.write.format("delta").mode("overwrite").save("delta_lake/clean_transactions")
alerts.write.format("delta").mode("overwrite").save("delta_lake/fraud_alerts")

# Save for Power BI / dashboard
alerts.select("transaction_id","user_id","amount","merchant","fraud_score") \
      .write.csv("output/fraud_alerts_today", header=True, mode="overwrite")

print("Fraud Detection Complete!")
print(f"Total transactions processed: {df.count()}")
print(f"Fraudulent transactions flagged: {alerts.count()} (saved to output/fraud_alerts_today)")

spark.stop()
