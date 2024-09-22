import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from matplotlib import pyplot as plt
from pandas import DataFrame


class EDAAnalyzer:

    def __init__(self, spark_session):
        self.dataframe = spark_session.dataframe
        self.spark = spark_session.spark

    def display_head(self, n=5):
        pandas_df = self.dataframe.limit(n).toPandas()
        return pandas_df

    def display_shape(self):
        df_rows = self.dataframe.count()
        df_cols = len(self.dataframe.columns)
        return f"Shape of data: rows: {df_rows}, cols: {df_cols}"

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

            # Append column information to the list
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
                'max_repeats': max_repeats
            })

        # Convert the list to a pandas DataFrame
        column_info_df = pd.DataFrame(column_info)
        column_info_df = column_info_df.sort_values(by='Percent Non-null', ascending=False).reset_index(drop=True)

        return column_info_df

    def get_top_n_repeated_values(self, column_name: str, n: int = None):

        column_dtype = dict(self.dataframe.dtypes)[column_name]
        is_array_column = column_dtype.startswith('array')

        if is_array_column:
            df_exploded = self.dataframe.withColumn(column_name, F.explode(F.col(column_name)))
            df_values = df_exploded.select(column_name)
        else:
            df_values = self.dataframe.select(column_name)

        df_grouped = df_values.groupBy(column_name).count()
        df_grouped = df_grouped.withColumn("diagnosis_code", F.col(f"{column_name}.diagnosis_code"))
        df_grouped = df_grouped.withColumn("diagnosis_code_length", F.length(F.col("diagnosis_code")))

        df_top_n = df_grouped.orderBy(F.desc("count"))
        if n:
            df_top_n = df_top_n.limit(n)
        pandas_df = df_top_n.toPandas()
        return pandas_df

    def get_fill_counts_for_unique_values(self, column_name: str):

        unique_values = [row[column_name] for row in self.dataframe.select(column_name).distinct().collect()]
        results = []

        for unique_value in unique_values:
            if unique_value is None:
                filtered_df = self.dataframe.filter(F.col(column_name).isNull())
            else:
                filtered_df = self.dataframe.filter(F.col(column_name) == unique_value)

            type_count = filtered_df.count()

            for col in self.dataframe.columns:
                if col == column_name:
                    continue

                non_null_count = filtered_df.filter(F.col(col).isNotNull()).count()
                percent_non_null = (non_null_count / type_count) * 100 if type_count > 0 else 0

                results.append({
                    'Unique Value': unique_value if unique_value is not None else 'None',
                    'Column': col,
                    'type_count': type_count,
                    'Non-null Count': non_null_count,
                    'Percent Non-null': percent_non_null
                })

        results_df = pd.DataFrame(results)
        return results_df

    def plot_percentile_based_cutoff(self, column_name: str, percentile: int = 90, cutoff_length: int = None):

        df_top_n = self.get_top_n_repeated_values(column_name, 1000)

        if cutoff_length:
            df_top_n = df_top_n[df_top_n['diagnosis_code_length']>cutoff_length]
        percentile_threshold = np.percentile(df_top_n['count'], percentile)

        plt.figure(figsize=(10, 6))
        plt.hist(df_top_n['count'], bins=30, color='skyblue', alpha=0.7)
        plt.axvline(percentile_threshold, color='red', linestyle='dashed', linewidth=2,
                    label=f'{percentile}th Percentile = {percentile_threshold}')
        plt.title(f'Distribution of Repeated Values and {percentile}th Percentile-Based Cutoff for {column_name}')
        plt.xlabel('Count (Number of Repeats)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    def convert_columns_to_float(self, columns: list):

        df = self.dataframe
        for column in columns:
            df = df.withColumn(column, F.col(column).cast('float'))
            print(f"Casted {column} to float")
        self.dataframe = df

    def display_column(self, column_name: str) -> DataFrame:

        column_df = self.dataframe.select(column_name)
        return column_df.toPandas()
