import json

# =========================================================
# Phase 1 / Phase 2 Tools
# =========================================================

from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler
from tools.operation_executor import OperationExecutor

# =========================================================
# Phase 3 Tools
# =========================================================

from tools.target_manager import TargetManager
from tools.feature_profiler import FeatureProfiler
from tools.feature_executor import FeatureExecutor

# =========================================================
# Phase 4 Tools
# =========================================================

from tools.model_profiler import ModelProfiler
from tools.model_executor import ModelExecutor

# =========================================================
# Phase 5 Tools
# =========================================================

from tools.training_executor import TrainingExecutor

# =========================================================
# Phase 6 Tools
# =========================================================

from tools.tuning_executor import TuningExecutor

# =========================================================
# Agents
# =========================================================

from agents.analysis_agent import AnalysisAgent
from agents.validation_agent import ValidationAgent

from agents.feature_engineering_agent import (
    FeatureEngineeringAgent
)

from agents.feature_validation_agent import (
    FeatureValidationAgent
)

from agents.model_selection_agent import (
    ModelSelectionAgent
)

from agents.model_validation_agent import (
    ModelValidationAgent
)

from agents.evaluation_agent import (
    EvaluationAgent
)

from agents.tuning_evaluation_agent import (
    TuningEvaluationAgent
)


def main():

    # =====================================================
    # AutoML Studio
    # =====================================================

    print("=" * 60)
    print("                 AutoML Studio")
    print("=" * 60)

    dataset_path = input(
        "Enter Dataset Path: "
    )

    # =====================================================
    # PHASE 1 : LOAD DATASET
    # =====================================================

    loader = DataLoader()

    df = loader.load_csv(
        dataset_path
    )

    print(
        "\nDataset Loaded Successfully."
    )

    # =====================================================
    # PHASE 1 : DATASET PROFILING
    # =====================================================

    profiler = DataProfiler(
        df
    )

    report = profiler.generate_report()

    # =====================================================
    # PHASE 1 : AI DATASET ANALYSIS
    # =====================================================

    print(
        "\nGenerating AI Recommendations...\n"
    )

    analysis_agent = AnalysisAgent()

    recommendations = analysis_agent.analyze(
        report
    )

    print(
        recommendations
    )

    # =====================================================
    # PHASE 2 : USER VALIDATION
    # =====================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "Review the recommendations above."
    )

    print(
        "Example:"
    )

    print(
        "Keep Cabin."
    )

    print(
        "Drop PassengerId."
    )

    print(
        "Also remove Ticket."
    )

    print(
        "=" * 60
    )

    user_response = input(
        "\nYour Decisions:\n\n"
    )

    validation_agent = ValidationAgent()

    operations = validation_agent.generate_operations(
        report,
        recommendations,
        user_response
    )

    if operations is None:

        print(
            "\nFailed to understand user response."
        )

        return

    if (
        not isinstance(operations, dict)
        or "operations" not in operations
    ):

        print(
            "\nInvalid preprocessing operations format."
        )

        return

    print(
        "\nGenerated Operations\n"
    )

    print(
        json.dumps(
            operations,
            indent=4
        )
    )

    # =====================================================
    # PHASE 2 : EXECUTE CLEANING
    # =====================================================

    executor = OperationExecutor(
        df
    )

    cleaned_df = executor.execute(
        operations["operations"]
    )

    executor.save_dataset(
        "data/processed/cleaned_dataset.csv"
    )

    print(
        "\nCleaning Completed Successfully."
    )

    print(
        "\nOperations Performed\n"
    )

    for log in executor.get_logs():

        print(log)

    # =====================================================
    # PHASE 3 : TARGET SELECTION
    # =====================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FEATURE ENGINEERING PHASE"
    )

    print(
        "=" * 60
    )

    target = input(
        "\nEnter Target Column : "
    )

    manager = TargetManager(
        cleaned_df
    )

    if not manager.validate_target(
        target
    ):

        print(
            "\nInvalid Target Column."
        )

        return

    problem_type = manager.get_problem_type(
        target
    )

    print(
        f"\nDetected Problem Type : "
        f"{problem_type}"
    )

    # =====================================================
    # PHASE 3 : FEATURE PROFILING
    # =====================================================

    feature_profiler = FeatureProfiler(
        dataframe=cleaned_df,
        target_column=target
    )

    feature_report = (
        feature_profiler.generate_report()
    )

    # =====================================================
    # PHASE 3 : AI FEATURE ENGINEERING
    # =====================================================

    feature_agent = (
        FeatureEngineeringAgent()
    )

    feature_recommendations = (
        feature_agent.analyze(
            feature_report,
            problem_type
        )
    )

    print("\n")

    print(
        feature_recommendations
    )

    # =====================================================
    # PHASE 3 : USER VALIDATION
    # =====================================================

    feature_user_response = input(
        "\nYour Decisions:\n\n"
    )

    feature_validator = (
        FeatureValidationAgent()
    )

    feature_operations = (
        feature_validator.generate_operations(
            feature_report,
            problem_type,
            feature_recommendations,
            feature_user_response
        )
    )

    if feature_operations is None:

        print(
            "\nFailed to understand "
            "feature engineering instructions."
        )

        return

    if (
        not isinstance(feature_operations, dict)
        or "operations" not in feature_operations
    ):

        print(
            "\nInvalid feature engineering "
            "operations format."
        )

        return

    print(
        "\nGenerated Feature Operations\n"
    )

    print(
        json.dumps(
            feature_operations,
            indent=4
        )
    )

    # =====================================================
    # PHASE 3 : EXECUTE FEATURE ENGINEERING
    # =====================================================

    feature_executor = FeatureExecutor(
        cleaned_df
    )

    final_df = feature_executor.execute(
        feature_operations["operations"]
    )

    feature_executor.save_dataset(
        "data/processed/"
        "feature_engineered_dataset.csv"
    )

    print(
        "\nFeature Engineering "
        "Completed Successfully."
    )

    print(
        "\nOperations Performed\n"
    )

    for log in feature_executor.get_logs():

        print(log)

    print("\n")

    print(
        "=" * 60
    )

    print(
        "AutoML Studio Phase 3 "
        "Completed Successfully."
    )

    print(
        "=" * 60
    )

    # =====================================================
    # PHASE 4 : MODEL PROFILING
    # =====================================================

    model_profiler = ModelProfiler(
        dataframe=final_df,
        target_column=target,
        problem_type=problem_type
    )

    model_report = (
        model_profiler.generate_report()
    )

    # =====================================================
    # PHASE 4 : AI MODEL SELECTION
    # =====================================================

    model_selection_agent = (
        ModelSelectionAgent()
    )

    model_recommendations = (
        model_selection_agent.analyze(
            model_report
        )
    )

    print("\n")

    print(
        model_recommendations
    )

    # =====================================================
    # PHASE 4 : USER MODEL VALIDATION
    # =====================================================

    model_user_response = input(
        "\nYour Decisions:\n\n"
    )

    model_validator = (
        ModelValidationAgent()
    )

    # IMPORTANT:
    # ModelValidationAgent expects:
    # problem_type,
    # recommendations,
    # user_response

    selected_models = (
        model_validator.generate_models(
            problem_type,
            model_recommendations,
            model_user_response
        )
    )

    if selected_models is None:

        print(
            "\nFailed to generate "
            "model selection."
        )

        return

    # =====================================================
    # PHASE 4 : EXTRACT MODEL LIST
    # =====================================================

    if isinstance(
        selected_models,
        dict
    ):

        selected_model_list = (
            selected_models.get(
                "models",
                []
            )
        )

    elif isinstance(
        selected_models,
        list
    ):

        selected_model_list = (
            selected_models
        )

    else:

        print(
            "\nInvalid model selection format."
        )

        return

    if not isinstance(
        selected_model_list,
        list
    ):

        print(
            "\nModel selection must be a list."
        )

        return

    if not selected_model_list:

        print(
            "\nNo models were selected."
        )

        return

    # =====================================================
    # PHASE 4 : MODEL EXECUTION
    # =====================================================

    model_executor = ModelExecutor()

    model_executor.execute(
        selected_model_list
    )

    print(
        "\nSelected Models\n"
    )

    for model_name in (
        model_executor.get_models()
    ):

        print(
            model_name
        )

    # =====================================================
    # PHASE 5 : MODEL TRAINING
    # =====================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "MODEL TRAINING & EVALUATION"
    )

    print(
        "=" * 60
    )

    training_executor = TrainingExecutor(
        dataframe=final_df,
        target_column=target,
        problem_type=problem_type
    )

    training_results = (
        training_executor.train_models(
            model_executor.get_models()
        )
    )

    if not training_results:

        print(
            "\nNo models were successfully trained."
        )

        return

    # =====================================================
    # PHASE 5 : DISPLAY BASELINE METRICS
    # =====================================================

    baseline_metrics = {

        model_name:
            result["metrics"]

        for model_name, result
        in training_results.items()

    }

    print("\n")

    print(
        "=" * 60
    )

    print(
        "BASELINE MODEL METRICS"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            baseline_metrics,
            indent=4
        )
    )

    # =====================================================
    # PHASE 5 : AI MODEL EVALUATION
    # =====================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "AI MODEL EVALUATION"
    )

    print(
        "=" * 60
    )

    evaluation_agent = (
        EvaluationAgent()
    )

    evaluation_result = (
        evaluation_agent.analyze(
            problem_type,
            baseline_metrics
        )
    )

    if evaluation_result is None:

        print(
            "\nModel evaluation failed."
        )

        return

    print(
        json.dumps(
            evaluation_result,
            indent=4
        )
    )

    # =====================================================
    # PHASE 5 : GET LLM SELECTED MODEL
    # =====================================================

    selected_model = (
        evaluation_result[
            "selected_model"
        ]
    )

    print(
        "\nSelected Model for "
        "Hyperparameter Optimization:"
    )

    print(
        selected_model
    )

    # =====================================================
    # PHASE 5 : SAVE BASELINE MODELS
    # =====================================================

    saved_baselines = (
        training_executor.save_models()
    )

    if not saved_baselines:

        print(
            "\nFailed to save baseline models."
        )

        return

    print(
        "\nBaseline Models Saved Successfully."
    )

    print("\n")

    print(
        "=" * 60
    )

    print(
        "AutoML Studio Phase 5 "
        "Completed Successfully."
    )

    print(
        "=" * 60
    )

    # =====================================================
    # PHASE 6 : GET SELECTED BASELINE RESULT
    # =====================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "HYPERPARAMETER OPTIMIZATION"
    )

    print(
        "=" * 60
    )

    baseline_result = (
        training_executor.get_model_result(
            selected_model
        )
    )

    if baseline_result is None:

        print(
            "\nSelected baseline model "
            "result not found."
        )

        return

    # =====================================================
    # PHASE 6 : GRID SEARCH
    # =====================================================

    tuning_executor = TuningExecutor(
        problem_type
    )

    tuning_result = (
        tuning_executor.tune_model(
            model_name=selected_model,
            baseline_result=baseline_result
        )
    )

    if tuning_result is None:

        print(
            "\nHyperparameter Optimization failed."
        )

        return

    # =====================================================
    # PHASE 6 : DISPLAY GRID SEARCH RESULTS
    # =====================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "GRID SEARCH RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        "\nBest Parameters:\n"
    )

    print(
        json.dumps(
            tuning_result[
                "best_parameters"
            ],
            indent=4
        )
    )

    print(
        "\nBest Cross-Validation Score:"
    )

    print(
        tuning_result[
            "best_cv_score"
        ]
    )

    print(
        "\nOptimization Metric:"
    )

    print(
        tuning_result[
            "scoring_metric"
        ]
    )

    # =====================================================
    # PHASE 6 : BASELINE VS TUNED METRICS
    # =====================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "BASELINE VS TUNED METRICS"
    )

    print(
        "=" * 60
    )

    metric_comparison = {

        "Baseline": (
            tuning_result[
                "baseline_metrics"
            ]
        ),

        "Tuned": (
            tuning_result[
                "tuned_metrics"
            ]
        )

    }

    print(
        json.dumps(
            metric_comparison,
            indent=4
        )
    )

    # =====================================================
    # PHASE 6 : AI TUNING EVALUATION
    # =====================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "AI OPTIMIZATION EVALUATION"
    )

    print(
        "=" * 60
    )

    tuning_evaluation_agent = (
        TuningEvaluationAgent()
    )

    tuning_evaluation = (
        tuning_evaluation_agent.analyze(
            problem_type,
            tuning_result
        )
    )

    if tuning_evaluation is None:

        print(
            "\nTuning evaluation failed."
        )

        return

    print(
        json.dumps(
            tuning_evaluation,
            indent=4
        )
    )

    # =====================================================
    # PHASE 6 : FINAL MODEL DECISION
    # =====================================================

    final_decision = (
        tuning_evaluation[
            "final_decision"
        ]
    )

    print(
        "\nFinal Model Decision:"
    )

    print(
        final_decision
    )

    # =====================================================
    # PHASE 6 : SAVE FINAL MODEL
    # =====================================================

    saved = (
        tuning_executor.save_final_model(
            final_decision
        )
    )

    if not saved:

        print(
            "\nFailed to save final model."
        )

        return

    print(
        "\nFinal Model Saved Successfully."
    )

    # =====================================================
    # COMPLETE
    # =====================================================

    print("\n")

    print(
        "=" * 60
    )

    print(
        "AutoML Studio Phase 6 "
        "Completed Successfully."
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()
