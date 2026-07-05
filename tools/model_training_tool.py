from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.neighbors import (
    KNeighborsClassifier
)

from sklearn.svm import (
    SVC,
    SVR
)

from xgboost import (
    XGBClassifier,
    XGBRegressor
)


class ModelTrainingTool:
    """
    Responsible for

    1. Splitting the dataset
    2. Creating ML models
    3. Training models
    4. Making predictions

    No evaluation is performed here.
    """

    def __init__(

        self,

        dataframe,

        target_column,

        problem_type

    ):

        self.df = dataframe

        self.target = target_column

        self.problem_type = problem_type

    # =====================================================
    # Train Test Split
    # =====================================================

    def split_data(self):

        X = self.df.drop(
            columns=[self.target]
        )

        y = self.df[self.target]

        return train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y if self.problem_type == "Classification" else None

        )

    # =====================================================
    # Create Model
    # =====================================================

    def get_model(self, model_name):

        if model_name == "LogisticRegression":

            return LogisticRegression(
                max_iter=1000,
                random_state=42
            )

        elif model_name == "DecisionTree":

            return DecisionTreeClassifier(
                random_state=42
            )

        elif model_name == "RandomForest":

            return RandomForestClassifier(
                random_state=42
            )

        elif model_name == "XGBoost":

            return XGBClassifier(

                random_state=42,

                eval_metric="logloss"

            )

        elif model_name == "SVM":

            return SVC(

                probability=True,

                random_state=42

            )

        elif model_name == "KNN":

            return KNeighborsClassifier()

        elif model_name == "LinearRegression":

            return LinearRegression()

        elif model_name == "DecisionTreeRegressor":

            return DecisionTreeRegressor(
                random_state=42
            )

        elif model_name == "RandomForestRegressor":

            return RandomForestRegressor(
                random_state=42
            )

        elif model_name == "XGBoostRegressor":

            return XGBRegressor(
                random_state=42
            )

        elif model_name == "SVR":

            return SVR()

        return None

    # =====================================================
    # Train Model
    # =====================================================

    def train_model(

        self,

        model_name

    ):

        model = self.get_model(model_name)

        if model is None:

            return None

        X_train, X_test, y_train, y_test = self.split_data()

        model.fit(

            X_train,

            y_train

        )

        y_pred = model.predict(
            X_test
        )

        y_prob = None

        if self.problem_type == "Classification":

            if hasattr(

                model,

                "predict_proba"

            ):

                y_prob = model.predict_proba(
                    X_test
                )[:, 1]

        return {
    # Name of the trained model
    "model_name": model_name,

    # Trained baseline model
    "model": model,

    # Training data
    # Required by GridSearchCV in Phase 6
    "X_train": X_train,
    "y_train": y_train,

    # Untouched test data
    # Used only for final evaluation
    "X_test": X_test,
    "y_test": y_test,

    # Baseline predictions
    "y_pred": y_pred,

    # Baseline probability predictions
    "y_prob": y_prob
}