from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler
from agents.analysis_agent import AnalysisAgent


def main():

    print("=" * 50)
    print("          AutoML Studio")
    print("=" * 50)

    dataset_path = input("Enter Dataset Path: ")

    loader = DataLoader()

    df = loader.load_csv(dataset_path)

    print("\nDataset Loaded Successfully.\n")

    profiler = DataProfiler(df)

    print("Analyzing Dataset...\n")

    report = profiler.generate_report()

    print("Generating AI Conclusions...\n")

    agent = AnalysisAgent()

    analysis = agent.analyze(report)

    print("=" * 50)

    print(analysis)

    print("=" * 50)


if __name__ == "__main__":
    main()