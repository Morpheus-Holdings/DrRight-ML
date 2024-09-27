from modules.feature_engineering import FeatureEngineer
from modules.spark_session import SparkManager

mx_submits_path = "./data_sample/mx_submits.parquet/"
mx_submits_line_path = "./data_sample/mx_submitsline.parquet/"


mx_submits_spark_manager = SparkManager(mx_submits_path)
mx_submits_fe = FeatureEngineer(mx_submits_spark_manager)
mx_submits_fe.train_autoencoder()