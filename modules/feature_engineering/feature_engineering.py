import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import seaborn as sns
from keras import layers
from pandas import DataFrame
from pyspark.errors import IllegalArgumentException
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql import Window
from pyspark.sql.functions import col, when, mean, lit
from pyspark.sql.types import FloatType, IntegerType
from pyspark.sql.types import StringType, DoubleType, DateType, ArrayType
from tensorflow import keras
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
        numeric_cols = [field.name for field in self.dataframe.schema.fields if
                        isinstance(field.dataType, (FloatType, IntegerType, DoubleType))]
        date_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, DateType)]
        array_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, ArrayType)]

        for col_name in string_cols:
            self.dataframe = self.dataframe.withColumn(
                col_name,
                when(col(col_name).isNull() | (col(col_name) == ''), lit('unknown')).otherwise(col(col_name))
            )

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

    def preprocess_data(self, exclude_cols=None):

        if exclude_cols is None:
            exclude_cols = []
        self.impute_missing_values()

        string_cols = [field.name for field in self.dataframe.schema.fields if
                       isinstance(field.dataType, StringType) and field.name not in exclude_cols]
        date_cols = [field.name for field in self.dataframe.schema.fields if
                     isinstance(field.dataType, DateType) and field.name not in exclude_cols]
        numeric_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, (
            FloatType, IntegerType, DoubleType)) and field.name not in exclude_cols]

        for date_col in date_cols:
            if date_col in self.dataframe.columns:
                print(f"Processing date column: {date_col}")
                self.dataframe = self.dataframe.withColumn(f"{date_col}_year", F.year(col(date_col)).cast(DoubleType())) \
                    .withColumn(f"{date_col}_month", F.month(col(date_col)).cast(DoubleType())) \
                    .withColumn(f"{date_col}_day", F.dayofmonth(col(date_col)).cast(DoubleType()))

        for string_col in string_cols:
            index_col = f"{string_col}_index"
            try:
                indexer = StringIndexer(inputCol=string_col, outputCol=index_col).fit(self.dataframe)
                self.dataframe = indexer.transform(self.dataframe)
            except Exception as e:
                print(f"Error in StringIndexer for {string_col} due to error: {e}")

        for col in string_cols:
            col_index = f"{col}_index"
            col_ohe = f"{col}_ohe"

            if col_index in self.dataframe.columns:
                try:
                    onehot_encoder = OneHotEncoder(inputCols=[col_index], outputCols=[col_ohe], handleInvalid="keep")
                    self.dataframe = onehot_encoder.fit(self.dataframe).transform(self.dataframe)
                    print(f"One-Hot Encoding applied successfully to column: {col}")

                except IllegalArgumentException as e:

                    print(f"Error applying OneHotEncoder to column: {col}")
                    print(f"Error message: {str(e)}")

        ohe_columns = [f"{col}_ohe" for col in string_cols if f"{col}_ohe" in self.dataframe.columns]
        feature_cols = ohe_columns + numeric_cols + \
                       [f"{date_col}_year" for date_col in date_cols] + \
                       [f"{date_col}_month" for date_col in date_cols] + \
                       [f"{date_col}_day" for date_col in date_cols]

        print(f"Assembling all features into a vector with {len(feature_cols)} columns.")

        missing_cols = [col for col in feature_cols if col not in self.dataframe.columns]
        if missing_cols:
            print(f"Warning: The following columns are missing and will be excluded: {missing_cols}")

        assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
        self.dataframe = assembler.transform(self.dataframe)

        print("Preprocessing complete. Feature vector created.")

    def build_autoencoder(self):
        input_dim = self.dataframe.select('features').head()[0].size

        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(600, activation='relu')(inputs)
        encoded = layers.Dense(400, activation='relu')(encoded)
        decoded = layers.Dense(600, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)

        self.autoencoder = keras.Model(inputs, decoded)
        self.encoder = keras.Model(inputs, encoded)

    def train_autoencoder(self, epochs: int = 50, batch_size: int = 256):
        self.build_autoencoder()

        def data_generator(batch_size):

            feature_rdd = self.dataframe.select('features').rdd.map(lambda row: row[0].toArray())
            batch = []

            for row in feature_rdd.toLocalIterator():
                batch.append(row)
                if len(batch) == batch_size:

                    yield np.array(batch), np.array(batch)
                    batch = []

            if batch:
                yield np.array(batch), np.array(batch)

        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        steps_per_epoch = (self.dataframe.count() + batch_size - 1) // batch_size

        self.autoencoder.fit(
            data_generator(batch_size),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch
        )

    def plot_feature_importance_heatmap(self):

        original_feature_names = [column for column in self.dataframe.columns if column != 'features']
        original_features_df = self.dataframe.select(original_feature_names).toPandas()

        original_features_df = original_features_df.apply(pd.to_numeric, errors='coerce')
        combined_df = pd.concat([original_features_df, self.encoded_dataframe], axis=1)

        numeric_combined_df = combined_df.select_dtypes(include=[float, int])
        num_original_features = len(original_feature_names)
        correlation_matrix = numeric_combined_df.corr().iloc[:num_original_features, num_original_features:]

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
        plt.title("Feature Importance Heatmap")
        plt.show()

    def get_distinct_values(self, column_name):

        try:

            if column_name in self.dataframe.columns:

                distinct_values = self.dataframe.select(column_name).distinct().collect()

                values_list = [row[column_name] for row in distinct_values]
                return values_list
            else:
                raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        except Exception as e:
            print(f"Error fetching distinct values from column '{column_name}': {e}")
            return None

    def add_comorbidities_with_exponential_decay_sparse_vector(self, decay_rate: float = 0.01):

        df = self.dataframe
        window_spec = Window.partitionBy('patient_id').orderBy('claim_statement_from_date') \
            .rowsBetween(Window.unboundedPreceding, -1)

        df = df.withColumn(
            'diagnosis_code_date',
            F.expr("""
                transform(
                    claim_all_diagnosis_codes, 
                    x -> struct(x.diagnosis_code as diagnosis_code, claim_statement_from_date as diagnosis_date)
                )
            """)
        )

        df = df.withColumn(
            'previous_comorbidities_with_dates',
            F.array_distinct(F.flatten(F.collect_list('diagnosis_code_date').over(window_spec)))
        )

        distinct_diagnosis_codes = df.select(F.explode('claim_all_diagnosis_codes').alias('diagnosis')) \
            .select('diagnosis.diagnosis_code') \
            .distinct() \
            .rdd.map(lambda row: row['diagnosis_code']) \
            .collect()

        code_to_index = {code: idx for idx, code in enumerate(distinct_diagnosis_codes)}

        def compute_decay(current_claim_date, diagnosis_dates):
            decay_values = {}

            for diagnosis in diagnosis_dates:
                diagnosis_code = diagnosis['diagnosis_code']
                diagnosis_date = diagnosis['diagnosis_date']

                if diagnosis_date is not None:

                    time_difference = (current_claim_date - diagnosis_date).days
                    decay_value = np.exp(-decay_rate * time_difference)

                    if diagnosis_code in decay_values:
                        decay_values[diagnosis_code] += decay_value
                    else:
                        decay_values[diagnosis_code] = decay_value

            indices = []
            values = []

            for code in distinct_diagnosis_codes:
                if code in decay_values:
                    indices.append(code_to_index[code])
                    values.append(decay_values[code])
                else:
                    indices.append(code_to_index[code])
                    values.append(0.0)

            return SparseVector(len(distinct_diagnosis_codes), indices, values)

        compute_decay_udf = F.udf(compute_decay, VectorUDT())

        df = df.withColumn(
            'previous_diagnosis_ohe',
            compute_decay_udf(F.col('claim_statement_from_date'), F.col('previous_comorbidities_with_dates'))
        )

        self.dataframe = df

        self.print_shape("DataFrame After Adding Previous Diagnosis OHE with Exponential Decay", self.dataframe)
