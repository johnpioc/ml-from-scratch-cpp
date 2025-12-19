import pandas as pd
import statsmodels.api as sm
import time


def main() -> None:
    # Load Data
    df: pd.DataFrame = pd.read_csv("data/Boston.csv", sep=",")
    df.drop(df.columns[0], axis=1, inplace=True)

    train_x: pd.Series = df["crim"]
    train_y: pd.Series = df["medv"]
    train_x = sm.add_constant(train_x)

    # Fit Linear Regression Model and output R Squared Value
    start = time.perf_counter()

    model = sm.OLS(train_y, train_x)
    results = model.fit()

    end = time.perf_counter()
    runtime = end - start
    print(f"Python statsmodels runtime: {(runtime * 1000):.4f} Milliseconds")
    print(f"Python statsmodels R Squared: {results.rsquared:.2f}")

main()
