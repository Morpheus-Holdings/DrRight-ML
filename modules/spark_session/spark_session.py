import os
import sys
from pyspark.sql import SparkSession


class SparkManager:
    def __init__(self, file_path):
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

        self.spark = SparkSession.builder \
            .appName("DB") \
            .config("spark.executor.memory", "8g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.memory", "8g") \
            .config("spark.sql.optimizer.maxIterations", "200") \
            .config("spark.sql.shuffle.partitions", "16") \
            .config("spark.sql.autoBroadcastJoinThreshold", "10485760") \
            .config("spark.sql.mapKeyDedupPolicy", "LAST_WIN") \
            .getOrCreate()

        self.dataframe = self.spark.read.parquet(file_path)
        self.dataframe = self.dataframe.repartition(200)
