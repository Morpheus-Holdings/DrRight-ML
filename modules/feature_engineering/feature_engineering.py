import pyspark.sql.functions as F
from pandas import DataFrame
from pyspark.sql import Window
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, DateType, IntegerType

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

    
    def add_procedure_array(self, procedure_column, date_column):
        df = self.dataframe
        self.print_shape(f"Initial DataFrame for {procedure_column}", df)
    
        # Define a window specification
        window_spec = Window.partitionBy('patient_id').orderBy('claim_statement_from_date')\
            .rowsBetween(Window.unboundedPreceding, -1)
    
        # Step 1: Add an index to preserve the original order of the rows
        df = df.withColumn(
            'row_num',
            F.row_number().over(Window.partitionBy('patient_id').orderBy(F.lit(1)))
        )
    
        # Step 2: Collect the procedure and date as structs over the window
        df = df.withColumn(
            f'{procedure_column}_date_struct',
            F.struct(
                F.col(procedure_column).alias('procedure'),
                F.col(date_column).alias('date'),
                F.col('row_num').alias('order')
            )
        )
    
        # Step 3: Define the UDF to keep only the latest occurrence of each procedure, while preserving the order
        def unique_procedures_with_latest_date(procedure_array):
            procedure_dict = {}
            procedure_order = {}
            for item in procedure_array:
                procedure = item['procedure']
                date = item['date']
                order = item['order']
                # Only keep the procedure with the latest date
                if procedure not in procedure_dict or procedure_dict[procedure] < date:
                    procedure_dict[procedure] = date
                    procedure_order[procedure] = order
            # Sort the procedures by their update order and return the list of {procedure, date}
            sorted_procedures = sorted(procedure_dict.items(), key=lambda x: procedure_order[x[0]])
            return [(procedure, date) for procedure, date in sorted_procedures]
    
        # Create the UDF
        schema = ArrayType(StructType([
            StructField("procedure", StringType(), True),
            StructField("date", DateType(), True)
        ]))
    
        unique_procedures_udf = F.udf(unique_procedures_with_latest_date, schema)
    
        # Step 4: Apply collect_list directly and the UDF in a single step
        df = df.withColumn(
            f'updated_{procedure_column}_array',
            unique_procedures_udf(
                F.collect_list(f'{procedure_column}_date_struct').over(window_spec)
            )
        )
    
        # Step 5: Drop temporary columns
        df = df.drop('row_num', f'{procedure_column}_date_struct')
    
        self.print_shape(f"DataFrame After Procedure Update for {procedure_column}", df)
    
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
