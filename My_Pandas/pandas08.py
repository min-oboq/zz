import pandas as pd
from pandas04 import pdprint
import seaborn as sns


def main():
    df = sns.load_dataset("titanic")
    print(df.info())
    print(df.head())
    
    print(df["who"].value_counts())
    print(df["pclass"].value_counts())
    print(df["pclass"].astype("int8").head())
    df["pclass"] = df["pclass"].astype("int8")
    print(df.info())
    
    print(df.sort_values("age").head())
    print(df.sort_values("age", ascending = False).head())
    
    cond = df["age"] >= 70
    print(df.loc[cond][["age", "who", "alive"]].head())


if __name__ == "__main__":
    main()