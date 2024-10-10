import os
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
import tensorflow as tf


class FeatureEngineer:

    def get_python_version(self):
        import sys
        return sys.version

    def __init__(self, spark_manager):
        self.feature_name_map = None
        self.code_to_index_length = None
        self.ohe_mapping_length = None
        self.feature_length = None
        self.ohe_mapping = {}
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
        limited_df = self.dataframe.limit(n).coalesce(1)
        pandas_df = limited_df.toPandas()
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
        self.numeric_cols = [field.name for field in self.dataframe.schema.fields if
                             isinstance(field.dataType, (FloatType, IntegerType, DoubleType))]
        date_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, DateType)]
        array_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, ArrayType)]

        for col_name in string_cols:
            self.dataframe = self.dataframe.withColumn(
                col_name,
                when(col(col_name).isNull() | (col(col_name) == ''), lit('unknown')).otherwise(col(col_name))
            )

        for col_name in self.numeric_cols:
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
        self.numeric_cols = [field.name for field in self.dataframe.schema.fields if isinstance(field.dataType, (
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

                mappings = indexer.labels
                for idx, value in enumerate(mappings):
                    value = value.replace(",", "")
                    self.ohe_mapping[f"{string_col}_{value}_index"] = idx

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

        self.ohe_columns = [f"{col}_ohe" for col in string_cols if f"{col}_ohe" in self.dataframe.columns]

        self.feature_cols = self.ohe_columns + self.numeric_cols + \
                            ['previous_diagnosis_ohe'] + \
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

    def get_ohe_mapping(self):
        return self.ohe_mapping

    def get_ohe_mapping_length(self):
        if not self.ohe_mapping_length:
            self.ohe_mapping_length = len(self.ohe_mapping)
        return self.ohe_mapping_length

    def get_code_to_index(self):
        return self.code_to_index

    def get_code_to_index_length(self):
        if not self.code_to_index_length:
            self.code_to_index_length = len(self.code_to_index)
        return self.code_to_index_length

    def get_feature_length(self):
        if not self.feature_length:
            self.feature_length = self.dataframe.select('features').head()[0].size
        return self.feature_length

    def build_autoencoder(self):

        self.feature_length = self.dataframe.select('features').head()[0].size
        print(f"input_features : {self.feature_length}")

        inputs = keras.Input(shape=(self.feature_length,))
        # encoded = layers.Dense(200, activation='relu')(inputs)
        encoded = layers.Dense(100, activation='relu')(inputs)
        # decoded = layers.Dense(200, activation='relu')(encoded)
        decoded = layers.Dense(self.feature_length, activation='sigmoid')(encoded)

        self.autoencoder = keras.Model(inputs, decoded)
        self.encoder = keras.Model(inputs, encoded)

    def extract_features_to_array(self, features):
        if features is not None:
            return features.toArray().tolist()
        return None

    def train_autoencoder(self, epochs: int = 50, batch_size: int = 256):

        self.dataframe.cache()
        self.dataframe = self.dataframe.select("features")

        strategy = tf.distribute.MirroredStrategy()

        with strategy.scope():
            input_dim = self.dataframe.head()["features"].size
            inputs = keras.Input(shape=(input_dim,))
            encoded = keras.layers.Dense(100, activation='relu')(inputs)
            decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

            self.autoencoder = keras.Model(inputs, decoded)
            self.encoder = keras.Model(inputs, encoded)

            self.autoencoder.compile(optimizer='adam', loss='huber_loss')

        def spark_to_dataset(dataframe, batch_size):
            def generator():
                for row in dataframe.toLocalIterator():
                    features = row["features"]
                    yield np.array(features.toArray()), np.array(features.toArray())

            dataset = tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),
                )
            )
            dataset = dataset.batch(batch_size).repeat()
            return dataset

        tf_dataset = spark_to_dataset(self.dataframe, batch_size)
        tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        df_rows = self.dataframe.count()
        steps_per_epoch = df_rows // batch_size
        print(f"Approximate distinct count: {df_rows}")
        print(f"Steps per epoch: {steps_per_epoch}")

        history = self.autoencoder.fit(
            tf_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=2
        )

        self.dataframe.unpersist()
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

        self.dataframe = self.dataframe.withColumn(
            'claim_all_diagnosis_codes_flat',
            F.expr("transform(claim_all_diagnosis_codes, x -> x.diagnosis_code)")
        )

        distinct_diagnosis_codes_df = self.dataframe.select(
            F.explode('claim_all_diagnosis_codes_flat').alias('diagnosis_code')
        ).distinct()

        distinct_diagnosis_codes = [row['diagnosis_code'] for row in distinct_diagnosis_codes_df.collect()]

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
        self.code_to_index = code_to_index
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

        distinct_diagnosis_codes_df = self.dataframe.select(
            F.explode('claim_all_diagnosis_codes_flat').alias('diagnosis_code')
        ).distinct()

        distinct_diagnosis_codes = [row['diagnosis_code'] for row in distinct_diagnosis_codes_df.collect()]

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

    from pyspark.sql import DataFrame

    def get_sorted_feature_correlations(self):

        encoder_output = self.encoder.transform(self.dataframe)
        features_df = encoder_output.select(
            self.feature_cols + [col for col in encoder_output.columns if col.startswith("encoded_")])

        correlation_matrix = features_df.stat.corr()

        avg_correlation_list = []

        for feature in self.feature_cols:
            # Calculate the average correlation of each feature with all encoded features
            avg_corr = correlation_matrix[feature][len(self.feature_cols):].mean()
            avg_correlation_list.append((feature, avg_corr))

        # Create a DataFrame from the average correlations
        avg_correlation_df = self.spark.createDataFrame(avg_correlation_list,
                                                        ["Feature", "Average Correlation with Dense Layer"])

        # Sort the DataFrame by average correlation in descending order
        sorted_features_df = avg_correlation_df.orderBy(F.col("Average Correlation with Dense Layer").desc())

        # Return the sorted DataFrame
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

    def display_shape(self):
        df_rows = self.dataframe.count()
        df_cols = len(self.dataframe.columns)
        return f"Shape of data: rows: {df_rows}, cols: {df_cols}"

    def save_autoencoder(self, save_dir: str):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        autoencoder_save_path = os.path.join(save_dir, "autoencoder.h5")
        self.autoencoder.save(autoencoder_save_path)
        print(f"Autoencoder model saved to {autoencoder_save_path}")

        encoder_save_path = os.path.join(save_dir, "encoder.h5")
        self.encoder.save(encoder_save_path)
        print(f"Encoder model saved to {encoder_save_path}")

    def load_autoencoder(self, load_dir: str):

        autoencoder_load_path = os.path.join(load_dir, "autoencoder.h5")
        encoder_load_path = os.path.join(load_dir, "encoder.h5")

        self.autoencoder = keras.models.load_model(autoencoder_load_path)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        print(f"Autoencoder model loaded and recompiled from {autoencoder_load_path}")

        self.encoder = keras.models.load_model(encoder_load_path)
        print(f"Encoder model loaded from {encoder_load_path}")

    def expand_features(self, selected_columns):

        feature_cols = self.feature_cols

        for i, feature_name in enumerate(feature_cols):

            if feature_name == 'previous_diagnosis_ohe':
                for diagnosis_code, index in self.code_to_index.items():
                    diagnosis_col = f"Diagnosis_{diagnosis_code}"
                    if diagnosis_col in selected_columns:
                        extract_diagnosis = F.udf(lambda v: float(v[index]) if v is not None else None, FloatType())
                        self.dataframe = self.dataframe.withColumn(diagnosis_col,
                                                                   extract_diagnosis(F.col('features')))
                        print(f"Created column for diagnosis code: {diagnosis_code} (index: {index})")
            else:

                for ohe_category, index in self.ohe_mapping.items():

                    if f"{feature_name}".replace("_ohe", "") == "_".join(ohe_category.split("_")[:-2]):
                        ohe_col = ohe_category.replace("_index", "")
                        if ohe_category == "principal_diagnosis_category_Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism_index":
                            print(f"ohe_col : {ohe_col}" )

                        if ohe_col in selected_columns:
                            extract_ohe = F.udf(lambda v: float(v[index]) if v is not None else 0.0, FloatType())
                            self.dataframe = self.dataframe.withColumn(ohe_col, extract_ohe(F.col(feature_name)))
                            print(f"Created OHE column: {ohe_col} (index: {index})")
                else:
                    if feature_name in selected_columns:
                        extract_feature = F.udf(lambda v: float(v[i]) if v is not None else None, FloatType())
                        self.dataframe = self.dataframe.withColumn(feature_name, extract_feature(F.col('features')))
                        print(f"Created column for feature: {feature_name}")

    def evaluate_feature_impact(self, start_index=None, end_index=None, batch_size=500):

        num_features = self.get_feature_length()

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = num_features

        input_data = np.zeros((1, num_features))

        original_output = self.autoencoder.predict(input_data)

        changes = {}

        for i in range(start_index, end_index, batch_size):

            batch_end = min(i + batch_size, end_index)
            batch_size_current = batch_end - i

            modified_batch = np.tile(input_data, (batch_size_current, 1))
            for j in range(batch_size_current):
                modified_batch[j, i + j] = 1

            batch_output = self.autoencoder.predict(modified_batch)

            for j in range(batch_size_current):
                feature_index = i + j
                change = np.mean(np.abs(batch_output[j] - original_output))
                changes[feature_index] = change

        changes_df = pd.DataFrame(list(changes.items()), columns=['Feature Index', 'Impact'])

        changes_df['Feature Name'] = changes_df['Feature Index'].map(self.feature_name_map)
        changes_df = changes_df.sort_values(by='Impact', ascending=False).reset_index(drop=True)

        return changes_df

    def evaluate_diagnosis_impact(self, diagnosis_code=None, start_index=None, end_index=None):
        num_features = self.get_feature_length()

        if diagnosis_code is not None:
            if diagnosis_code not in self.code_to_index:
                print(f"Diagnosis code {diagnosis_code} doesn't exist")
                return
            else:

                start_index = self.get_ohe_mapping_length() + len(self.numeric_cols) + self.code_to_index[
                    diagnosis_code]
                end_index = start_index + 1

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = num_features

        input_data = np.zeros((1, num_features))
        original_output = self.autoencoder.predict(input_data)

        changes = {}

        for feature_index in range(start_index, end_index):
            modified_data_1 = np.copy(input_data)
            modified_data_1[:, feature_index] = 1

            output_1 = self.autoencoder.predict(modified_data_1)
            change = np.mean(np.abs(output_1 - original_output))
            changes[feature_index] = change

        changes_df = pd.DataFrame(list(changes.items()), columns=['Feature Index', 'Impact'])

        changes_df['Feature Name'] = changes_df['Feature Index'].map(self.feature_name_map)
        changes_df = changes_df.sort_values(by='Impact', ascending=False).reset_index(drop=True)

        return changes_df

    def create_feature_name_map(self):

        feature_name_map = {}
        index_counter = 0

        for col, ohe_indices in self.ohe_mapping.items():
            feature_name_map[index_counter] = f"{col}"
            index_counter += 1

        for col in self.numeric_cols:
            feature_name_map[index_counter] = col
            index_counter += 1

        for diagnosis, idx in self.code_to_index.items():
            feature_name_map[index_counter + idx] = f"Diagnosis_{diagnosis}"

        self.feature_name_map = feature_name_map

    def transform_line_level_procedure_codes(self):

        self.dataframe = self.dataframe.withColumn(
            'line_level_procedure_codes_flat',
            F.expr("transform(servicelines, x -> x.line_level_procedure_code)")
        )

        distinct_line_level_procedure_codes = self.dataframe.select(
            F.explode('servicelines').alias('procedure')
        ).select('procedure.line_level_procedure_code').distinct().rdd.map(
            lambda row: row['line_level_procedure_code']).collect()

        self.update_code_to_index(distinct_line_level_procedure_codes)

        code_to_index = self.code_to_index

        def generate_sparse_vector(procedure_codes):
            if procedure_codes is None:
                size = len(code_to_index)
                return SparseVector(size, [], [])

            unique_codes = set(procedure_codes)
            indices = [code_to_index.get(code) for code in unique_codes if code in code_to_index]
            size = len(code_to_index)

            return SparseVector(size, sorted(indices), [1.0] * len(indices))

        sparse_vector_udf = F.udf(generate_sparse_vector, VectorUDT())

        self.dataframe = self.dataframe.withColumn(
            'line_level_procedures_ohe',
            sparse_vector_udf(F.col('line_level_procedure_codes_flat'))
        )

        return self.display_top_rows_as_pandas("line_level_procedures_ohe")

    def display_column_info(self):

        # Initialize a list to hold the column information
        column_info = []
        total_rows = self.dataframe.count()

        # Get the first row of the DataFrame
        first_row = self.dataframe.limit(1).collect()
        if first_row:
            first_row = first_row[0].asDict()
        else:
            first_row = {}

        # Iterate over each column in the DataFrame
        for col, dtype in self.dataframe.dtypes:
            non_null_count = self.dataframe.filter(F.col(col).isNotNull()).count()
            percent_non_null = (non_null_count / total_rows) * 100

            sample_value = first_row.get(col, None)
            most_frequent = self.dataframe.groupBy(col).count().orderBy(F.desc('count')).first()
            most_frequent_value = most_frequent[col]
            max_repeats = most_frequent['count']
            distinct_count = self.dataframe.select(col).distinct().count()

            if dtype in ['int', 'double', 'float']:
                # For numeric columns
                min_value = self.dataframe.agg(F.min(col)).collect()[0][0]
                max_value = self.dataframe.agg(F.max(col)).collect()[0][0]

            elif dtype == 'string':
                # For string columns
                min_value = self.dataframe.agg(F.min(F.length(col))).collect()[0][0]
                max_value = self.dataframe.agg(F.max(F.length(col))).collect()[0][0]
            elif dtype.startswith('date'):
                # For date columns
                min_value = self.dataframe.agg(F.min(col)).collect()[0][0]
                max_value = self.dataframe.agg(F.max(col)).collect()[0][0]
            else:
                # For other types
                min_value = None
                max_value = None
                max_repeats = None

            column_info.append({
                'Column Name': col,
                'Non-null Count': non_null_count,
                'Percent Non-null': percent_non_null,
                'Min Value': min_value,
                'Max Value': max_value,
                'Max Repeats': max_repeats,
                'Sample': sample_value,
                'Data Type': dtype,
                'most_frequent_value': most_frequent_value,
                'max_repeats': max_repeats,
                'distinct_count': distinct_count
            })

        # Convert the list to a pandas DataFrame
        column_info_df = pd.DataFrame(column_info)
        column_info_df = column_info_df.sort_values(by='Percent Non-null', ascending=False).reset_index(drop=True)

        return column_info_df

    def preprocess_features(self, feature_columns, label_column):

        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        self.dataframe = assembler.transform(self.dataframe)
        self.dataframe = self.dataframe.select("features", label_column)
        return self.dataframe