from typing import List

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F


class KMeansBuilder:
    def __init__(self, model_data):
        self.dataframe = model_data
        self.model = None
        self.evaluator = ClusteringEvaluator()
        self.train_df, self.test_df = model_data.split_data(train_ratio=0.8)

    def fit_model(self, k: int):
        kmeans = KMeans(k=k, seed=42, featuresCol='features')
        self.model = kmeans.fit(self.train_df)
        return self.model

    def evaluate_model(self, type="Train") -> dict:
        if self.model is None:
            raise Exception("Model has not been fitted yet.")

        if type == "Train":
            predictions = self.model.transform(self.train_df)
        else:
            predictions = self.model.transform(self.test_df)

        inertia = self.model.summary.trainingCost
        silhouette_score = self.evaluator.evaluate(predictions)

        return {
            "Train : inertia": inertia,
            "silhouette_score": silhouette_score
        }

    def optimal_k(self, clusters=None) -> list[float]:

        wssse = []
        for k in clusters:
            kmeans = KMeans(k=k, seed=42, featuresCol='features')
            model = kmeans.fit(self.dataframe)
            wssse.append(model.summary.trainingCost)

        optimal_k = np.argmin(np.diff(np.diff(wssse)))
        print(f"Optimal k  : {clusters[optimal_k]}")
        return wssse

    def get_model_summary(self):
        if self.model is None:
            raise Exception("Model has not been fitted yet.")
        return self.model.summary

    def save_model(self, path: str):
        if self.model is None:
            raise Exception("Model has not been fitted yet.")
        self.model.save(path)

    def load_model(self, path: str):
        from pyspark.ml.clustering import KMeansModel
        self.model = KMeansModel.load(path)

    def get_cluster_averages(self, column: str, dataset_type: str = 'train'):

        if self.model is None:
            raise Exception("Model has not been fitted yet.")

        if dataset_type == "Train":
            predictions = self.model.transform(self.train_df)
        else:
            predictions = self.model.transform(self.test_df)

        cluster_averages = predictions.groupBy("prediction").agg(F.avg(F.col(column)).alias(f"{column}_avg"))

        print(f"Cluster averages for {column} in the {dataset_type} set:")
        cluster_averages.show()

        return cluster_averages
