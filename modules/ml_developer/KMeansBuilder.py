from typing import List

import pandas as pd
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
        self.train_df, self.test_df = model_data.randomSplit([0.8, 0.2], seed=42)

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

    def optimal_k(self, clusters=None) -> list[dict]:

        wssse = []
        for k in clusters:
            kmeans = KMeans(k=k, seed=42, featuresCol='features')
            model = self.fit_model(k)
            sc = self.evaluate_model()
            wssse.append(sc)
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

    def get_cluster_averages(self, column: str) -> pd.DataFrame:
        if self.model is None:
            raise Exception("Model has not been fitted yet.")

        train_predictions = self.model.transform(self.train_df)
        train_avg = train_predictions.groupBy("prediction").agg(
            F.avg(F.col(column)).alias(f"{column}_avg"),
            F.count("*").alias("train_size")
        )

        test_predictions = self.model.transform(self.test_df)
        test_avg = test_predictions.groupBy("prediction").agg(
            F.avg(F.col(column)).alias(f"{column}_avg"),
            F.count("*").alias("test_size")
        )

        train_avg_pd = train_avg.toPandas()
        test_avg_pd = test_avg.toPandas()

        merged_df = pd.merge(train_avg_pd, test_avg_pd, on='prediction', suffixes=('_train', '_test'))

        merged_df['difference'] = merged_df[f"{column}_avg_train"] - merged_df[f"{column}_avg_test"]
        merged_df['percent_diff'] = (merged_df['difference'] / merged_df[f"{column}_avg_train"].replace(0, np.nan)) * 100

        merged_df.rename(columns={'prediction': 'cluster_number'}, inplace=True)

        merged_df.sort_values(by='percent_diff', ascending=False, inplace=True)

        print(f"Cluster averages for {column}:")
        print(merged_df)

        return merged_df

    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        if self.model is None:
            raise Exception("Model has not been fitted yet.")

        centroids = self.model.clusterCenters()

        centroids_df = pd.DataFrame(centroids, columns=feature_names)

        mean_importance = centroids_df.abs().mean()
        std_dev_importance = centroids_df.std()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_importance': mean_importance,
            'std_dev_importance': std_dev_importance
        })

        importance_df.sort_values(by='mean_importance', ascending=False, inplace=True)

        print("Feature Importance based on cluster centroids:")
        print(importance_df)

        return importance_df