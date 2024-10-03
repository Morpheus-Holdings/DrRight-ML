from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import seaborn as sns
from keras import layers
from pandas import DataFrame
from pyspark.errors import IllegalArgumentException
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.linalg import SparseVector, VectorUDT, DenseVector
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
        self.code_to_index = {}
        self.feature_cols = None
        self.spark = spark_manager.spark
        self.dataframe = spark_manager.dataframe
        self.index_counter = 0

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

        self.dataframe = df_filtered

    def remove_procedure_codes(self, procedure_list):
        df = self.dataframe

        df_exploded = df.withColumn("exploded_procedure", F.explode(F.col("servicelines")))

        df_filtered = df_exploded.filter(~F.col("exploded_procedure.line_level_procedure_code").isin(procedure_list))

        df_filtered = df_filtered.groupBy([col for col in df.columns if col != "servicelines"]).agg(
            F.collect_list("exploded_procedure").alias("servicelines")
        )

        self.print_shape("DataFrame After Removing Procedure Codes", df_filtered)
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

        # for col_name in array_cols:
        #     self.dataframe = self.dataframe.withColumn(col_name,
        #                                                when(col(col_name).isNull(), lit([])).otherwise(col(col_name)))

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
        # ['previous_diagnosis_ohe'] + \
        self.feature_cols = ohe_columns + numeric_cols + \
                            [f"{date_col}_year" for date_col in date_cols] + \
                            [f"{date_col}_month" for date_col in date_cols] + \
                            [f"{date_col}_day" for date_col in date_cols]

        print(f"Assembling all features into a vector with {len(self.feature_cols)} columns.")

        missing_cols = [col for col in self.feature_cols if col not in self.dataframe.columns]
        if missing_cols:
            print(f"Warning: The following columns are missing and will be excluded: {missing_cols}")

        assembler = VectorAssembler(inputCols=self.feature_cols, outputCol='features')
        self.dataframe = assembler.transform(self.dataframe)

        print("Preprocessing complete. Feature vector created.")

    def build_autoencoder(self):

        input_dim = self.dataframe.select('features').head()[0].size
        print(f"input_features : {input_dim}")

        inputs = keras.Input(shape=(input_dim,))
        # encoded = layers.Dense(200, activation='relu')(inputs)
        encoded = layers.Dense(100, activation='relu')(inputs)
        # decoded = layers.Dense(200, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = keras.Model(inputs, decoded)
        self.encoder = keras.Model(inputs, encoded)

    def extract_features_to_array(self, features):
        if features is not None:
            return features.toArray().tolist()
        return None

    def train_autoencoder(self, epochs: int = 50, batch_size: int = 256):

        self.build_autoencoder()

        self.autoencoder.compile(optimizer='adam', loss='huber_loss')

        self.dataframe.cache()

        steps_per_epoch = self.dataframe.count() // batch_size

        def data_generator(batch_size=256):
            while True:
                batches = []
                for row in self.dataframe.collect():
                    dense_vector = row.features.toArray()
                    dense_vector = dense_vector.reshape(1, -1)
                    batches.append(dense_vector)
                    if len(batches) == batch_size:
                        batch_data = np.vstack(batches)
                        yield batch_data, batch_data
                        batches = []
                if batches:
                    batch_data = np.vstack(batches)
                    yield batch_data, batch_data

        history = self.autoencoder.fit(
            data_generator(batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=2
        )

        return history

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

        def compute_decay(claim_date, previous_comorbidities):
            if previous_comorbidities is None or claim_date is None:
                return SparseVector(len(code_to_index), [], [])

            decay_dict = defaultdict(float)

            for diagnosis_code, date in previous_comorbidities:
                if date is not None:
                    if diagnosis_code in code_to_index:
                        index = code_to_index[diagnosis_code]
                        decay_factor = decay_rate ** (claim_date - date).days
                        decay_dict[index] += decay_factor

            indices = sorted(decay_dict.keys())
            values = [decay_dict[idx] for idx in indices]

            return SparseVector(len(code_to_index), indices, values)

        compute_decay_udf = F.udf(compute_decay, VectorUDT())

        df = df.withColumn(
            'previous_diagnosis_ohe',
            compute_decay_udf(F.col('claim_statement_from_date'), F.col('previous_comorbidities_with_dates'))
        )

        self.dataframe = df
        return self.display_top_rows_as_pandas("previous_diagnosis_ohe")

    def retain_columns(self, column_list: list):

        existing_columns = self.dataframe.columns
        missing_columns = [col for col in column_list if col not in existing_columns]

        if missing_columns:
            raise ValueError(f"The following columns do not exist in the DataFrame: {missing_columns}")

        self.dataframe = self.dataframe.select(column_list)

    def add_to_code_to_index(self, diagnosis_code):

        if diagnosis_code not in self.code_to_index:
            self.code_to_index[diagnosis_code] = self.index_counter
            self.index_counter += 1
        return self.code_to_index[diagnosis_code]

    def update_code_to_index(self, new_codes):

        for code in new_codes:
            self.add_to_code_to_index(code)

    def transform_claim_all_diagnosis_codes(self):

        self.dataframe = self.dataframe.withColumn(
            'claim_all_diagnosis_codes_flat',
            F.expr("transform(claim_all_diagnosis_codes, x -> x.diagnosis_code)")
        )

        distinct_diagnosis_codes = self.dataframe.select(
            F.explode('claim_all_diagnosis_codes').alias('diagnosis')
        ).select('diagnosis.diagnosis_code').distinct().rdd.map(lambda row: row['diagnosis_code']).collect()

        self.update_code_to_index(distinct_diagnosis_codes)

        code_to_index = self.code_to_index

        def generate_sparse_vector(diagnosis_codes):
            if diagnosis_codes is None:

                size = len(code_to_index)
                return SparseVector(size, [], [])

            unique_codes = set(diagnosis_codes)
            indices = [code_to_index.get(code) for code in unique_codes if code in code_to_index]
            size = len(code_to_index)

            return SparseVector(size, sorted(indices), [1.0] * len(indices))

        sparse_vector_udf = F.udf(generate_sparse_vector, VectorUDT())

        self.dataframe = self.dataframe.withColumn(
            'claim_all_diagnosis_ohe',
            sparse_vector_udf(F.col('claim_all_diagnosis_codes_flat'))
        )

        return self.display_top_rows_as_pandas("claim_all_diagnosis_ohe")

    def get_columns_as_pandas_df(self):
        columns = self.dataframe.columns
        return pd.DataFrame(columns, columns=['Column Names'])

    def get_feature_columns(self):
        return self.feature_cols

    def display_top_rows_as_pandas(self, column_name, n=5):

        if column_name not in self.dataframe.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        top_rows_df = self.dataframe.select(column_name).limit(n).toPandas()

        return top_rows_df

    def convert_columns_to_float(self, columns: list):

        df = self.dataframe
        for column in columns:
            df = df.withColumn(column, F.col(column).cast('float'))
            print(f"Casted {column} to float")
        self.dataframe = df

    def reduce_dataframe_size(self, target_size: int = 1000):
        self.dataframe = self.dataframe.limit(target_size)

    def plot_correlation_heatmap(self):

        features_df = self.dataframe.select('features').toPandas()
        features_df = features_df.reset_index(drop=True)

        features_df['features'] = features_df['features'].apply(
            lambda x: x.toArray() if isinstance(x, (SparseVector, DenseVector)) else x
        )

        features_array = np.stack(features_df['features'].values)

        try:
            encoder_output = self.encoder.predict(features_array)
        except Exception as e:
            print(f"Error during encoding: {e}")
            return

        encoder_df = pd.DataFrame(encoder_output, columns=[f"encoded_{i}" for i in range(encoder_output.shape[1])])

        combined_df = pd.concat([pd.DataFrame(features_array), encoder_df], axis=1)

        # combined_df = combined_df.corr().iloc[:len(features_array[0]), len(features_array[0]):]
        correlation_matrix = combined_df.corr()

        # Plot the heatmap
        # plt.figure(figsize=(12, 10))
        # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        # plt.title('Feature and Encoder Correlation Heatmap')
        # plt.tight_layout()
        # plt.show()

        return correlation_matrix

    def get_sorted_feature_correlations(self):

        features_df = self.dataframe.select('features').toPandas()
        features_df = features_df.reset_index(drop=True)

        features_df['features'] = features_df['features'].apply(
            lambda x: x.toArray() if isinstance(x, (SparseVector, DenseVector)) else x
        )

        features_array = np.stack(features_df['features'].values)

        try:
            encoder_output = self.encoder.predict(features_array)
        except Exception as e:
            print(f"Error during encoding: {e}")
            return None

        encoder_df = pd.DataFrame(encoder_output, columns=[f"encoded_{i}" for i in range(encoder_output.shape[1])])

        combined_df = pd.concat([pd.DataFrame(features_array), encoder_df], axis=1)

        correlation_matrix = combined_df.corr()

        correlation_with_dense = correlation_matrix.iloc[:-encoder_output.shape[1], -encoder_output.shape[1]:]

        avg_correlation = correlation_with_dense.mean(axis=1)

        sorted_features_df = avg_correlation.reset_index()
        sorted_features_df.columns = ['Feature', 'Average Correlation with Dense Layer']
        sorted_features_df = sorted_features_df.sort_values(by='Average Correlation with Dense Layer', ascending=False)

        return sorted_features_df

    def add_procedures_with_exponential_decay_sparse_vector(self, decay_rate: float = 0.01):

        df = self.dataframe
        window_spec = Window.partitionBy('patient_id').orderBy('claim_statement_from_date') \
            .rowsBetween(Window.unboundedPreceding, -1)

        df = df.withColumn(
            'line_level_procedure_code_date',
            F.expr("""
                transform(
                    servicelines, 
                    x -> struct(x.line_level_procedure_code as line_level_procedure_code, claim_statement_from_date as line_level_procedure_date)
                )
            """)
        )

        df = df.withColumn(
            'previous_procedures_with_dates',
            F.array_distinct(F.flatten(F.collect_list('line_level_procedure_code_date').over(window_spec)))
        )

        distinct_line_level_procedure_codes = df.select(F.explode('servicelines').alias('procedures')) \
            .select('procedures.line_level_procedure_code') \
            .distinct() \
            .rdd.map(lambda row: row['line_level_procedure_code']) \
            .collect()

        code_to_index = {code: idx for idx, code in enumerate(distinct_line_level_procedure_codes)}

        def compute_decay(current_claim_date, line_level_procedure_date):
            decay_values = {}

            for line_level_procedure in line_level_procedure_date:
                line_level_procedure_code = line_level_procedure['line_level_procedure_code']
                line_level_procedure_date = line_level_procedure['line_level_procedure_date']

                if line_level_procedure_date is not None:

                    time_difference = (current_claim_date - line_level_procedure_date).days
                    decay_value = np.exp(-decay_rate * time_difference)

                    if line_level_procedure_code in decay_values:
                        decay_values[line_level_procedure_code] += decay_value
                    else:
                        decay_values[line_level_procedure_code] = decay_value

            indices = []
            values = []

            for code in distinct_line_level_procedure_codes:
                if code in decay_values:
                    indices.append(code_to_index[code])
                    values.append(decay_values[code])
                else:
                    indices.append(code_to_index[code])
                    values.append(0.0)

            return SparseVector(len(distinct_line_level_procedure_codes), indices, values)

        compute_decay_udf = F.udf(compute_decay, VectorUDT())

        df = df.withColumn(
            'previous_line_level_procedure_ohe',
            compute_decay_udf(F.col('claim_statement_from_date'), F.col('previous_procedures_with_dates'))
        )

        self.dataframe = df

        self.print_shape("DataFrame After Adding Previous Procedures OHE with Exponential Decay", self.dataframe)
