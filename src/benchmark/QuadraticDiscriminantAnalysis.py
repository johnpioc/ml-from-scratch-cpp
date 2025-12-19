import pandas as pd 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import time

def main() -> None:
    # Load Data
    df: pd.DataFrame = pd.read_csv("data/SMarket.csv", sep=",")
    df.drop(df.columns[0], axis=1, inplace=True)

    # Train Model
    train_x: pd.Series = df[["Lag1", "Lag2"]]
    train_y: pd.Series = ((df["Direction"]) == "Down").astype(int)

    start = time.perf_counter()

    model = QuadraticDiscriminantAnalysis().fit(train_x, train_y)

    end = time.perf_counter()
    runtime = end - start 

    # Get Mean Error Rate
    estimates = pd.Series(model.predict(train_x))
    num_incorrect = (estimates != train_y).sum()
    error_rate = num_incorrect / train_y.size * 100
    print(f"Python sklearn Training Time: {(runtime * 1000):.4f} Milliseconds")
    print(f"Python sklearn Error Rate: {error_rate:.2f}%")

main()
