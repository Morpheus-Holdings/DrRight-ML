import pyspark.sql.functions as F


class FeatureEngineer:
    def __init__(self, spark_manager):
        self.spark = spark_manager.spark
        self.dataframe = spark_manager.dataframe

    def print_shape(self, message: str,df):
        print(f"{message} - Shape: {df.count()} rows, {len(df.columns)} columns")

    def add_comorbidities_array(self):
        df = self.dataframe
        self.print_shape("Initial DataFrame", df)

        df_previous = df.withColumnRenamed('claim_statement_from_date', 'previous_claim_date') \
            .withColumnRenamed('claim_all_diagnosis_codes', 'previous_diagnosis_codes')

        self.print_shape("DataFrame with Renamed Columns", df_previous)

        df_joined = df.alias('current') \
            .join(df_previous.alias('previous'),
                  (F.col('current.patient_id') == F.col('previous.patient_id')) &
                  (F.col('previous.previous_claim_date') < F.col('current.claim_statement_from_date')),
                  how='left') \
            .select('current.*', 'previous.previous_diagnosis_codes')

        self.print_shape("DataFrame After Join", df_joined)

        df_comorbidities = df_joined.groupBy('patient_id', 'claim_statement_from_date', 'claim_all_diagnosis_codes') \
            .agg(
            F.expr("array_distinct(flatten(collect_list(previous_diagnosis_codes)))").alias('previous_comorbidities'))

        self.print_shape("DataFrame After Aggregation", df_comorbidities)

        self.dataframe = df.alias('original') \
            .join(df_comorbidities.alias('comorbidities'),
                  (F.col('original.patient_id') == F.col('comorbidities.patient_id')) &
                  (F.col('original.claim_statement_from_date') == F.col('comorbidities.claim_statement_from_date')) &
                  (F.col('original.claim_all_diagnosis_codes') == F.col('comorbidities.claim_all_diagnosis_codes')),
                  how='left') \
            .select('original.*', 'comorbidities.previous_comorbidities')

        self.print_shape("DataFrame After Final Join", self.dataframe)

    def display_head(self, n=5):
        pandas_df = self.dataframe.limit(n).toPandas()
        return pandas_df