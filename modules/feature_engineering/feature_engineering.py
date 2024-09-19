import pyspark.sql.functions as F
from pandas import DataFrame
from pyspark.sql import Window


class FeatureEngineer:
    def __init__(self, spark_manager):
        self.spark = spark_manager.spark
        self.dataframe = spark_manager.dataframe

    def print_shape(self, message: str,df):
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

    def add_procedure_array(self):
        df = self.dataframe
        self.print_shape("Initial DataFrame", df)

        window_spec = Window.partitionBy('patient_id').orderBy('claim_statement_from_date') \
            .rowsBetween(Window.unboundedPreceding, -1)

        df = df.withColumn(
            'previous_procedure',
            F.array_distinct(F.collect_list('principal_procedure_code').over(window_spec))
        )

        self.print_shape("DataFrame After Window Function", df)

        self.dataframe = df

    def display_head(self, n=5):
        pandas_df = self.dataframe.limit(n).toPandas()
        return pandas_df

    def get_rows_by_column_value(self, column_name: str, value) -> DataFrame:
        return self.dataframe.filter(self.dataframe[column_name] == value).toPandas()