from pyspark.sql import SparkSession


class SparkManager:

    def __init__(self, file_path):
        self.spark = SparkSession.builder \
            .appName("DB") \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "2") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
            .getOrCreate()
        self.dataframe = self.spark.read.parquet(file_path)