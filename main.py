from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler
from tools.target_manager import TargetManager
from tools.feature_profiler import FeatureProfiler

from agents.analysis_agent import AnalysisAgent
from agents.validation_agent import ValidationAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.feature_validation_agent import FeatureValidationAgent

from tools.operation_executor import OperationExecutor
from tools.feature_executor import FeatureExecutor

import json


def main():

    print("=" * 60)
    print("                 AutoML Studio")
    print("=" * 60)

    dataset_path = input("Enter Dataset Path: ")

    # ======================================================
    # Phase 1 : Load Dataset
    # ======================================================

    loader = DataLoader()

    df = loader.load_csv(dataset_path)

    print("\nDataset Loaded Successfully.")

    # ======================================================
    # Phase 1 : Dataset Analysis
    # ======================================================

    profiler = DataProfiler(df)

    report = profiler.generate_report()

    print("\nGenerating AI Recommendations...\n")

    analysis_agent = AnalysisAgent()

    recommendations = analysis_agent.analyze(report)

    print(recommendations)

    # ======================================================
    # Phase 2 : User Validation
    # ======================================================

    print("\n" + "=" * 60)
    print("Review the recommendations above.")
    print("Example:")
    print("Keep Cabin.")
    print("Drop PassengerId.")
    print("Also remove Ticket.")
    print("=" * 60)

    user_response = input("\nYour Decisions:\n\n")

    validation_agent = ValidationAgent()

    operations = validation_agent.generate_operations(
        report,
        recommendations,
        user_response
    )

    if operations is None:

        print("\nFailed to understand user response.")

        return

    print("\nGenerated Operations\n")

    print(json.dumps(
        operations,
        indent=4
    ))

    # ======================================================
    # Phase 2 : Execute Cleaning
    # ======================================================

    executor = OperationExecutor(df)

    cleaned_df = executor.execute(
        operations["operations"]
    )

    executor.save_dataset(
        "data/processed/cleaned_dataset.csv"
    )

    print("\nCleaning Completed Successfully.")

    print("\nOperations Performed\n")

    for log in executor.get_logs():

        print(log)

    # ======================================================
    # Phase 3 : Target Selection
    # ======================================================

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PHASE")
    print("=" * 60)

    target = input(
        "\nEnter Target Column : "
    )

    manager = TargetManager(
        cleaned_df
    )

    if not manager.validate_target(target):

        print("\nInvalid Target Column.")

        return

    problem_type = manager.get_problem_type(
        target
    )

    print(f"\nDetected Problem Type : {problem_type}")

    # ======================================================
    # Phase 3 : Feature Profiling
    # ======================================================

    feature_profiler = FeatureProfiler(
        dataframe=cleaned_df,
        target_column=target
    )

    feature_report = feature_profiler.generate_report()

    # ======================================================
    # Phase 3 : AI Feature Engineering
    # ======================================================

    feature_agent = FeatureEngineeringAgent()

    feature_recommendations = feature_agent.analyze(
        feature_report,
        problem_type
    )

    print("\n")

    print(feature_recommendations)

    # ======================================================
    # Phase 3 : User Validation
    # ======================================================

    user_response = input(
        "\nYour Decisions:\n\n"
    )

    feature_validator = FeatureValidationAgent()

    feature_operations = feature_validator.generate_operations(
        feature_report,
        problem_type,
        feature_recommendations,
        user_response
    )
    if feature_operations is None:

        print("\nFailed to understand feature engineering instructions.")

        return

    print("\nGenerated Feature Operations\n")

    print(json.dumps(
        feature_operations,
        indent=4
    ))

    # ======================================================
    # Phase 3 : Execute Feature Engineering
    # ======================================================

    feature_executor = FeatureExecutor(
        cleaned_df
    )

    final_df = feature_executor.execute(
        feature_operations["operations"]
    )

    # Save Feature Engineered Dataset
    feature_executor.save_dataset(
        "data/processed/feature_engineered_dataset.csv"
    )

    print("\nFeature Engineering Completed Successfully.")

    print("\nOperations Performed\n")

    for log in feature_executor.get_logs():

        print(log)

    print("\n")
    print("=" * 60)
    print("AutoML Studio Phase 3 Completed Successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()