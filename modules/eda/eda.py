import pandas as pd
import pyspark.sql.functions as F


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

            # Initialize min_value, max_value, and max_repeats
            min_value = None
            max_value = None
            max_repeats = None
            sample_value = first_row.get(col, None)

            if dtype in ['int', 'double']:
                # For numeric columns
                min_value = self.dataframe.agg(F.min(col)).collect()[0][0]
                max_value = self.dataframe.agg(F.max(col)).collect()[0][0]
                max_repeats = self.dataframe.groupBy(col).count().agg(F.max('count')).collect()[0][0]
            elif dtype == 'string':
                # For string columns
                min_value = self.dataframe.agg(F.min(F.length(col))).collect()[0][0]
                max_value = self.dataframe.agg(F.max(F.length(col))).collect()[0][0]
                max_repeats = self.dataframe.groupBy(col).count().agg(F.max('count')).collect()[0][0]
            elif dtype.startswith('date'):
                # For date columns
                min_value = self.dataframe.agg(F.min(col)).collect()[0][0]
                max_value = self.dataframe.agg(F.max(col)).collect()[0][0]
                max_repeats = self.dataframe.groupBy(col).count().agg(F.max('count')).collect()[0][0]
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
                'Data Type': dtype
            })

        # Convert the list to a pandas DataFrame
        column_info_df = pd.DataFrame(column_info)
        column_info_df = column_info_df.sort_values(by='Percent Non-null', ascending=False).reset_index(drop=True)

        return column_info_df
