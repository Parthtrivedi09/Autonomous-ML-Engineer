from sklearn.metrics import (

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,

    mean_absolute_error,
    root_mean_squared_error,
    r2_score

)


class EvaluationTool:
    """
    Responsible for evaluating trained models.

    It NEVER trains models.

    It only computes evaluation metrics.
    """

    def __init__(

        self,

        problem_type

    ):

        self.problem_type = problem_type

    # =====================================================
    # Classification Metrics
    # =====================================================

    def classification_metrics(

        self,

        y_true,

        y_pred,

        y_prob=None

    ):

        metrics = {

            "Accuracy":
                round(
                    accuracy_score(
                        y_true,
                        y_pred
                    ),
                    4
                ),

            "Precision":
                round(
                    precision_score(
                        y_true,
                        y_pred,
                        zero_division=0
                    ),
                    4
                ),

            "Recall":
                round(
                    recall_score(
                        y_true,
                        y_pred,
                        zero_division=0
                    ),
                    4
                ),

            "F1 Score":
                round(
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0
                    ),
                    4
                )

        }

        if y_prob is not None:

            metrics["ROC AUC"] = round(

                roc_auc_score(

                    y_true,

                    y_prob

                ),

                4

            )

        return metrics

    # =====================================================
    # Regression Metrics
    # =====================================================

    def regression_metrics(

        self,

        y_true,

        y_pred

    ):

        return {

            "MAE":
                round(
                    mean_absolute_error(
                        y_true,
                        y_pred
                    ),
                    4
                ),

            "RMSE":
                round(
                    root_mean_squared_error(
                        y_true,
                        y_pred
                    ),
                    4
                ),

            "R2 Score":
                round(
                    r2_score(
                        y_true,
                        y_pred
                    ),
                    4
                )

        }

    # =====================================================
    # Main Evaluation Function
    # =====================================================

    def evaluate(

        self,

        y_true,

        y_pred,

        y_prob=None

    ):

        if self.problem_type == "Classification":

            return self.classification_metrics(

                y_true,

                y_pred,

                y_prob

            )

        return self.regression_metrics(

            y_true,

            y_pred

        )