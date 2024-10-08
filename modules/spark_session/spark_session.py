import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

class SparkManager:
    def __init__(self, file_path=None, cohort_key=None):
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        self.spark = SparkSession.builder \
            .appName("DB") \
            .config("spark.executor.memory", "8g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.memory", "8g") \
            .config("spark.sql.optimizer.maxIterations", "200") \
            .config("spark.network.timeout", "800s") \
            .config("spark.executor.heartbeatInterval", "60s") \
            .config("spark.sql.shuffle.partitions", "16") \
            .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
            .config("spark.sql.mapKeyDedupPolicy", "LAST_WIN") \
            .getOrCreate()

        if file_path:
            self.dataframe = self.spark.read.parquet(file_path)
        else:

            self.dataframe = self.spark.table("doctorright.asset.mx_submits")
            if cohort_key is not None:
                self.dataframe = self.dataframe.filter(col("cohort_key") == cohort_key)

        self.dataframe = self.dataframe.repartition(200)