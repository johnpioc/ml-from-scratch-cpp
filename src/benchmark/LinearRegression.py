import numpy as np
import pandas as pd
import statsmodels.api as sm


def main() -> None:
    # Load Data
    df: pd.DataFrame = pd.read_csv("data/Boston.csv", sep=",")
    df.drop(df.columns[0], axis=1, inplace=True)

    train_x: pd.DataFrame = df.drop(["indus", "age", "medv"], axis=1)
    train_y: pd.Series = df["medv"]
    train_x = sm.add_constant(train_x)

    # Fit Linear Regression Model and output R Squared Value
    model = sm.OLS(train_y, train_x)
    results = model.fit()
    print(results.summary())
    print(f"Python statsmodels R Squared: {results.rsquared:.2f}")

main()
