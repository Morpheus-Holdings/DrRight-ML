import os

import pyspark.sql.functions as F
from pyspark.ml.classification import MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Bucketizer
from pyspark.sql.types import IntegerType


class MLPModelBuilder:
    def __init__(self, model_data, feature_columns, label_column, model_name="MLP_model"):
        self.train_predictions = None
        self.test_predictions = None
        self.model_name = model_name
        self.mlp_model = None
        self.dataframe = model_data
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.train_df = None
        self.test_df = None
        self.bucketer = None
        self.max_claim_amount = model_data.select(F.max(label_column)).first()[0]

    def split_data(self, test_size=0.2, random_state=42):
        train_df, test_df = self.dataframe.randomSplit([1 - test_size, test_size], seed=random_state)
        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df

    def bin_labels(self, num_bins=3):

        percentiles = [float(i) / num_bins for i in range(num_bins + 1)]
        splits = self.dataframe.approxQuantile(self.label_column, percentiles, 0.0)

        self.bucketer = Bucketizer(splits=splits, inputCol=self.label_column, outputCol="label_binned")
        self.train_df = self.bucketer.transform(self.train_df)
        self.test_df = self.bucketer.transform(self.test_df)

    def train_model(self, layers):

        mlp = MultilayerPerceptronClassifier(
            layers=layers,
            labelCol="label_binned",
            featuresCol="features",
            maxIter=100,
            blockSize=128,
            seed=1234
        )

        self.mlp_model = mlp.fit(self.train_df)
        return self.mlp_model

    def evaluate_model(self, type="Test"):

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label_binned",
            predictionCol="prediction",
            metricName="accuracy"
        )

        if type == "Test":

            self.test_predictions = self.mlp_model.transform(self.test_df)
            accuracy = evaluator.evaluate(self.test_predictions)
        else:

            self.train_predictions = self.mlp_model.transform(self.train_df)
            accuracy = evaluator.evaluate(self.train_predictions)

        return accuracy

    def save_model(self, path=None):
        if self.mlp_model is not None:
            save_path = path if path else os.path.join(os.getcwd(), self.model_name)
            self.mlp_model.save(save_path)
            print(f"Model '{self.model_name}' saved to {save_path}")
        else:
            print("Model is not trained yet. Please train the model before saving.")

    @classmethod
    def load_model(cls, model_data, feature_columns, label_column, model_name="MLP_model", path=None):
        load_path = path if path else os.path.join(os.getcwd())
        model = MultilayerPerceptronClassificationModel.load(load_path)
        instance = cls(model_data, feature_columns, label_column, model_name=model_name)
        instance.mlp_model = model
        return instance

    def average_claim_by_bin_train(self):

        if self.train_predictions is None:
            self.train_predictions = self.mlp_model.transform(self.train_df)

        avg_claim_train = self.train_predictions.groupBy("label_binned").agg(
            F.avg(self.label_column).alias("average_claim_amount")
        )

        return avg_claim_train

    def average_claim_by_bin_test(self):
        if self.test_predictions is None:
            self.test_predictions = self.mlp_model.transform(self.test_df)

        avg_claim_test = self.test_predictions.groupBy("label_binned").agg(
            F.avg(self.label_column).alias("average_claim_amount")
        )

        return avg_claim_test

    def average_claim_by_predicted_bin_train(self, num_bins=3):
        if self.train_predictions is None:
            self.train_predictions = self.mlp_model.transform(self.train_df)

        # Calculate the bin size once
        bin_size = self.max_claim_amount / num_bins

        # Define UDF outside of the DataFrame operations
        def bin_function(x):
            return int(x // bin_size) if x is not None else None

        bin_udf = F.udf(bin_function, IntegerType())

        # Use the UDF to create the "predicted_bin" column
        self.train_predictions = self.train_predictions.withColumn("predicted_bin", bin_udf(F.col("prediction")))

        avg_claim_pred_train = self.train_predictions.groupBy("predicted_bin").agg(
            F.avg(self.label_column).alias("average_claim_amount")
        ).orderBy("predicted_bin")

        return avg_claim_pred_train

    def average_claim_by_predicted_bin_test(self, num_bins=3):
        if self.test_predictions is None:
            self.test_predictions = self.mlp_model.transform(self.test_df)

        # Calculate the bin size once
        bin_size = self.max_claim_amount / num_bins

        # Use a simple arithmetic operation to create the predicted_bin column
        self.test_predictions = self.test_predictions.withColumn(
            "predicted_bin",
            (F.col("prediction") / bin_size).cast(IntegerType())
        )

        avg_claim_pred_test = self.test_predictions.groupBy("predicted_bin").agg(
            F.avg(self.label_column).alias("average_claim_amount")
        ).orderBy("predicted_bin")

        return avg_claim_pred_test


