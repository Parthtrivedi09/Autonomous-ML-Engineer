from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler

from agents.analysis_agent import AnalysisAgent
from agents.validation_agent import ValidationAgent

from tools.operation_executor import OperationExecutor
import json


def main():

    print("=" * 50)
    print("           AutoML Studio")
    print("=" * 50)

    dataset_path = input("Enter Dataset Path: ")

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    loader = DataLoader()

    df = loader.load_csv(dataset_path)

    print("\nDataset Loaded Successfully.")

    # --------------------------------------------------
    # Generate Dataset Report
    # --------------------------------------------------

    profiler = DataProfiler(df)

    report = profiler.generate_report()

    # --------------------------------------------------
    # AI Analysis
    # --------------------------------------------------

    print("\nGenerating AI Recommendations...\n")

    analysis_agent = AnalysisAgent()

    recommendations = analysis_agent.analyze(report)

    print(recommendations)

    # --------------------------------------------------
    # User Validation
    # --------------------------------------------------

    print("\n" + "=" * 50)

    print("Review the recommendations above.")

    print("Example:")

    print("Keep Cabin.")

    print("Drop PassengerId.")

    print("Also remove Ticket.")

    print("=" * 50)

    user_response = input("\nYour Decisions:\n\n")

    # --------------------------------------------------
    # Validation Agent
    # --------------------------------------------------

    validation_agent = ValidationAgent()

    operations = validation_agent.generate_operations(
    report,
    recommendations,
    user_response
)

    if operations is None:

        print("\nFailed to understand user response.")

        return

    print("\nOperations Generated Successfully.\n")

    print(json.dumps(
    operations,
    indent=4
    ))

    # --------------------------------------------------
    # Execute Operations
    # --------------------------------------------------

    executor = OperationExecutor(df)

    executor.execute(operations["operations"])

    # --------------------------------------------------
    # Save Dataset
    # --------------------------------------------------

    executor.save_dataset(

        "data/processed/cleaned_dataset.csv"

    )

    print("\nDataset Saved Successfully.")

    # --------------------------------------------------
    # Show Log
    # --------------------------------------------------

    print("\nOperations Performed\n")

    for log in executor.get_logs():

        print(log)


if __name__ == "__main__":
    main()