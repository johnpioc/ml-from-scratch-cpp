import pandas as pd 
import statsmodels.api as sm
import time

def main() -> None:
    # Load Data
    df: pd.DataFrame = pd.read_csv("data/SMarket.csv", sep=",")
    df.drop(df.columns[0], axis=1, inplace=True)

    # Train Model
    train_x: pd.Series = df["Lag1"]
    train_y: pd.Series = ((df["Direction"]) == "Down").astype(int)

    start = time.perf_counter()

    model = sm.Logit(train_y, train_x)
    results = model.fit(disp=0)

    end = time.perf_counter()
    runtime = end - start

    # Get Estimates 
    estimates = pd.Series(map(round, results.predict(train_x)))

    # Get Mean Error Rate
    num_incorrect = (estimates != train_y).sum()
    error_rate = num_incorrect / train_y.size * 100
    print(f"Python statsmodels Training Time: {(runtime * 1000):.4f} Milliseconds")
    print(f"Python statsmodels Error Rate: {error_rate:.2f}%")

main()
