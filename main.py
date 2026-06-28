from tools.data_loader import DataLoader
from tools.data_profiler import DataProfiler


def main():

    loader = DataLoader()

    df = loader.load_csv("data/raw/titanic.csv")

    profiler = DataProfiler(df)

    report = profiler.generate_report()

    print(report)


if __name__ == "__main__":
    main()


