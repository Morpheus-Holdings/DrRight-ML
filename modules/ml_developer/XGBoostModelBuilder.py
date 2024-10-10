from matplotlib import pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from xgboost.spark import SparkXGBRegressor

class XGBoostModelBuilder:
    def __init__(self, model_data, feature_columns, label_column):
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

    def evaluate_model(self):

        predictions = self.xgb_model.transform(self.test_df)

        evaluator = RegressionEvaluator(
            label_col=self.label_column,
            prediction_col="prediction",
            metricName="rmse"
        )
        rmse = evaluator.evaluate(predictions)
        return rmse

    def feature_importance(self):

        feature_importances = self.xgb_model.nativeBooster.getScore(importance_type='weight')
        sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        return sorted_importances

    def plot_tree(self, model, num_trees=0, num_levels=3):
        booster = model.nativeBooster

        # Plot the tree for the specified number of levels
        plt.figure(figsize=(12, 8))
        self.xgb_model.plot_tree(booster, num_trees=num_trees, rankdir="UT", max_depth=num_levels)
        plt.show()