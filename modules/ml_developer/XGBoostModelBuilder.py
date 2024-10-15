import os

from pyspark.ml.evaluation import RegressionEvaluator
from xgboost.spark import SparkXGBRegressor, SparkXGBRegressorModel
import pyspark.sql.functions as F

class XGBoostModelBuilder:
    def __init__(self, model_data, feature_columns, label_column, model_name="XGB_model"):
        self.train_predictions = None
        self.test_predictions = None
        self.model_name = model_name
        self.xgb_model = None
        self.dataframe = model_data
        self.feature_columns = feature_columns
        self.label_column = label_column
        self.train_df = None
        self.test_df = None

    def split_data(self, test_size=0.2, random_state=42):
        train_df, test_df = self.dataframe.randomSplit([1 - test_size, test_size], seed=random_state)
        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df

    def train_model(self):

        xgb_regressor = SparkXGBRegressor(
            features_col="features",
            label_col=self.label_column,
            objective="reg:squarederror",
            max_depth=3,
            eta=0.1,
            num_round=100
        )

        self.xgb_model = xgb_regressor.fit(self.train_df)
        return self.xgb_model

    def evaluate_model(self, type="Test"):

        evaluator = RegressionEvaluator(
            labelCol=self.label_column,
            predictionCol="prediction",
            metricName="rmse"
        )

        if type == "Test":
            self.test_predictions = self.xgb_model.transform(self.test_df)
            rmse = evaluator.evaluate(self.test_predictions)
        else:
            self.train_predictions = self.xgb_model.transform(self.train_df)
            rmse = evaluator.evaluate(self.train_predictions)

        return rmse

    def feature_importance(self):

        booster = self.xgb_model.get_booster()
        feature_importances = booster.get_score(importance_type='weight')
        feature_mapping = {f"f{idx}": name for idx, name in enumerate(self.feature_columns)}

        feature_importance_with_names = {feature_mapping.get(k, k): v for k, v in feature_importances.items()}
        sorted_importances = sorted(feature_importance_with_names.items(), key=lambda x: x[1], reverse=True)

        return sorted_importances

    def calculate_accuracy(self, type="Test", metric="r2"):
        evaluator = RegressionEvaluator(
            labelCol=self.label_column,
            predictionCol="prediction",
            metricName=metric
        )

        if type == "Test":
            if self.test_predictions is None:
                self.test_predictions = self.xgb_model.transform(self.test_df)
            accuracy = evaluator.evaluate(self.test_predictions)
        else:
            if self.train_predictions is None:
                self.train_predictions = self.xgb_model.transform(self.train_df)
            accuracy = evaluator.evaluate(self.train_predictions)

        return accuracy

    def calculate_mape(self, type="Test"):

        if type == "Test":
            if self.test_predictions is None:
                self.test_predictions = self.xgb_model.transform(self.test_df)
            predictions_df = self.test_predictions
        else:
            if self.train_predictions is None:
                self.train_predictions = self.xgb_model.transform(self.train_df)
            predictions_df = self.train_predictions

        mape_df = predictions_df.withColumn(
            "absolute_percentage_error",
            F.abs((F.col(self.label_column) - F.col("prediction")) / F.col(self.label_column))
        )

        mape = mape_df.selectExpr("avg(absolute_percentage_error) as mape").collect()[0]["mape"]
        return mape * 100

    def save_model(self, path=None):
        if self.xgb_model is not None:

            save_path = path if path else os.path.join(os.getcwd(), self.model_name)
            self.xgb_model.save(save_path)
            print(f"Model '{self.model_name}' saved to {save_path}")
        else:
            print("Model is not trained yet. Please train the model before saving.")

    @classmethod
    def load_model(cls, model_data, feature_columns, label_column, model_name="XGB_model", path=None):
        load_path = path if path else os.path.join(os.getcwd())
        model = SparkXGBRegressorModel.load(load_path)
        instance = cls(model_data, feature_columns, label_column, model_name=model_name)
        instance.xgb_model = model
        return instance
