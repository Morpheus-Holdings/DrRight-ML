import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import seaborn as sns
from pandas import DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.sql import Window
from pyspark.sql.types import MapType, FloatType, IntegerType, StructType
from pyspark.sql.types import StringType, DoubleType, DateType, ArrayType
from tensorflow import keras
from pyspark.sql.functions import col, when, isnan, count, mean, udf, lit, coalesce
from tensorflow.keras import layers


class FeatureEngineer:

    def get_python_version(self):
        import sys
        return sys.version

    def __init__(self, spark_manager):
        self.encoder = None
        self.autoencoder = None
        self.encoded_dataframe = None
        self.encoded_features = None
        self.spark = spark_manager.spark
        self.dataframe = spark_manager.dataframe

    def print_shape(self, message: str, df):
        print(f"{message} - Shape: {df.count()} rows, {len(df.columns)} columns")

    def add_comorbidities_array(self):
        df = self.dataframe
        self.print_shape("Initial DataFrame", df)

        window_spec = Window.partitionBy('patient_id').orderBy('claim_statement_from_date') \
            .rowsBetween(Window.unboundedPreceding, -1)

        df = df.withColumn(
            'previous_comorbidities',
            F.array_distinct(F.flatten(F.collect_list('claim_all_diagnosis_codes').over(window_spec)))
        )

        self.print_shape("DataFrame After Window Function", df)

        self.dataframe = df

    def add_procedure_array(self, procedure_column, date_column):
        df = self.dataframe
        self.print_shape(f"Initial DataFrame for {procedure_column}", df)
    
        # Define a window specification that partitions by patient_id and orders by claim_statement_from_date
        window_spec = Window.partitionBy('patient_id').orderBy('claim_statement_from_date') \
            .rowsBetween(Window.unboundedPreceding, -1)
    
        # Filter out rows where either the procedure or date is null
        df = df.withColumn(
            'filtered_procedure',
            F.when(F.col(procedure_column).isNotNull() & F.col(date_column).isNotNull(), F.col(procedure_column))
        ).withColumn(
            'filtered_date',
            F.when(F.col(procedure_column).isNotNull() & F.col(date_column).isNotNull(), F.col(date_column))
        )
    
        # Create a map (procedure, date) pair
        df = df.withColumn(
            f'procedure_date_map',
            F.map_from_arrays(
                F.collect_list('filtered_procedure').over(window_spec), 
                F.collect_list('filtered_date').over(window_spec)
            )
        )
    
        # Define a UDF to keep the procedure with the latest date
        def update_procedure_map(procedure_date_map):
            if not procedure_date_map:
                return {}
            latest_map = {}
            for procedure, date in procedure_date_map.items():
                if procedure not in latest_map or date > latest_map[procedure]:
                    latest_map[procedure] = date
            return latest_map
    
        # Register the UDF with PySpark
        update_procedure_map_udf = F.udf(update_procedure_map, MapType(StringType(), StringType()))
    
        # Apply the UDF to update the procedure map with the latest date for each procedure
        df = df.withColumn(
            f'updated_procedure_date_map',
            update_procedure_map_udf(F.col(f'procedure_date_map'))
        )
    
        self.print_shape(f"DataFrame After Window Function for {procedure_column}", df)
    
        # Set the dataframe back to the instance
        self.dataframe = df

    def display_head(self, n=5):
        pandas_df = self.dataframe.limit(n).toPandas()
        return pandas_df

    def get_rows_by_column_value(self, column_name: str, value) -> DataFrame:
        return self.dataframe.filter(self.dataframe[column_name] == value).toPandas()

    def remove_diagnosis_codes(self, diagnosis_list):
        df = self.dataframe

        df_exploded = df.withColumn("exploded_diagnosis", F.explode(F.col("claim_all_diagnosis_codes")))

        df_filtered = df_exploded.filter(~F.col("exploded_diagnosis.diagnosis_code").isin(diagnosis_list))

        df_filtered = df_filtered.groupBy([col for col in df.columns if col != "claim_all_diagnosis_codes"]).agg(
            F.collect_list("exploded_diagnosis").alias("claim_all_diagnosis_codes")
        )

        self.print_shape("DataFrame After Removing Diagnosis Codes", df_filtered)
        self.dataframe = df_filtered

    def calculate_first_visit_and_duration(self):
        df = self.dataframe
        window_spec = Window.partitionBy('patient_id').orderBy('claim_statement_from_date')
        df = df.withColumn('first_visit_date', F.first('claim_statement_from_date').over(window_spec))
        df = df.withColumn('days_since_first_visit',
                           F.datediff(F.col('claim_statement_from_date'), F.col('first_visit_date')))

        self.print_shape("DataFrame After Calculating First Visit Date and Duration", df)
        self.dataframe = df

    def add_continuous_visit_years(self):

        df = self.dataframe

        df = df.withColumn('visit_year', F.year(F.col('claim_statement_from_date')))
        window_spec = Window.partitionBy('patient_id').orderBy('visit_year')
        df = df.withColumn('row_number', F.row_number().over(window_spec))
        df = df.withColumn('year_diff', F.col('visit_year') - F.col('row_number'))
        consecutive_window = Window.partitionBy('patient_id', 'year_diff').orderBy('visit_year')
        df = df.withColumn('continuous_visit_years', F.row_number().over(consecutive_window))
        self.dataframe = df

    def get_min_max(self, column_name: str):
        df = self.dataframe

        min_value = df.agg(F.min(column_name)).collect()[0][0]
        max_value = df.agg(F.max(column_name)).collect()[0][0]

        print(f"Min: {min_value}, Max: {max_value}")


    def add_train_test_indicator(self, test_size: float = 0.2) -> DataFrame:

        unique_patient_ids = self.dataframe.select('patient_id').distinct()
        test_patient_ids = unique_patient_ids.sample(False, test_size, seed=42).collect()

        test_patient_id_list = [row['patient_id'] for row in test_patient_ids]

        self.dataframe = self.dataframe.withColumn(
            'train_test',
            F.when(F.col('patient_id').isin(test_patient_id_list), 'test').otherwise('train')
        )

        return self.display_head()

    def impute_missing_values(self):
        string_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, StringType)]
        numeric_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, (FloatType, IntegerType, DoubleType))]
        date_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, DateType)]
        array_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, ArrayType)]

        for col_name in string_cols:
            self.dataframe = self.dataframe.withColumn(col_name, when(col(col_name).isNull(), lit('unknown')).otherwise(
                col(col_name)))

        for col_name in numeric_cols:
            mean_value = self.dataframe.select(mean(col(col_name))).first()[0]
            self.dataframe = self.dataframe.fillna({col_name: mean_value})

        for col_name in date_cols:
            self.dataframe = self.dataframe.withColumn(col_name,
                                                       when(col(col_name).isNull(), lit('1970-01-01')).otherwise(
                                                           col(col_name)))

        for col_name in array_cols:
            self.dataframe = self.dataframe.withColumn(col_name,
                                                       when(col(col_name).isNull(), lit([])).otherwise(col(col_name)))

    def preprocess_data(self):
        self.impute_missing_values()

        string_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, StringType)]
        date_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, DateType)]
        numeric_cols = [field.name for field in self.dataframe.schema.fields if
                        isinstance(field.dataType, (FloatType, IntegerType, DoubleType))]

        for date_col in date_cols:
            if date_col in self.dataframe.columns:
                self.dataframe = self.dataframe.withColumn(f"{date_col}_year", F.year(col(date_col)).cast(DoubleType())) \
                    .withColumn(f"{date_col}_month", F.month(col(date_col)).cast(DoubleType())) \
                    .withColumn(f"{date_col}_day", F.dayofmonth(col(date_col)).cast(DoubleType()))

        for column in string_cols:
            index_values = self.dataframe.select(column).distinct().rdd.flatMap(lambda x: x).collect()
            index_dict = {value: idx for idx, value in enumerate(index_values)}

            mapping_df = self.spark.createDataFrame(index_dict.items(), schema=["value", "index"])
            mapping_df = mapping_df.withColumnRenamed("value", column)

            self.dataframe = self.dataframe.join(mapping_df, on=column, how='left') \
                .withColumnRenamed("index", f"{column}_index")

        feature_cols = (
                [f"{column}_index" for column in string_cols] +
                numeric_cols +
                [f"{date_col}_year" for date_col in date_cols] +
                [f"{date_col}_month" for date_col in date_cols] +
                [f"{date_col}_day" for date_col in date_cols]
        )

        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        self.dataframe = assembler.transform(self.dataframe)

    def build_autoencoder(self):
        input_dim = self.dataframe.select('features').head()[0].size

        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(inputs)
        encoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

        self.autoencoder = keras.Model(inputs, decoded)
        self.encoder = keras.Model(inputs, encoded)

    def train_autoencoder(self, epochs: int = 50, batch_size: int = 256):

        self.build_autoencoder()
        feature_data = np.array(self.dataframe.select('features').rdd.map(lambda row: row[0].toArray()).collect())

        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        self.autoencoder.fit(feature_data, feature_data, epochs=epochs, batch_size=batch_size)

        self.encoded_features = self.encoder.predict(feature_data)
        self.encoded_dataframe = pd.DataFrame(self.encoded_features)

    def plot_feature_importance_heatmap(self):

        original_feature_names = [column for column in self.dataframe.columns if column != 'features']

        original_features_df = pd.DataFrame(
            np.array(self.dataframe.select(original_feature_names).collect()),
            columns=original_feature_names
        )
        encoded_features_df = self.encoded_dataframe

        correlation_matrix = pd.concat([original_features_df, encoded_features_df], axis=1).corr().iloc[
                             :len(original_feature_names), len(original_feature_names):]

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title('Correlation Heatmap: Original Features vs Encoded Features')
        plt.xlabel('Encoded Features')
        plt.ylabel('Original Features')
        plt.show()
