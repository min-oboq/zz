# author: choi sugil
# date: 2023.10.05 version: 1.0.0 license: MIT brief: keyward
# description: indexing
import numpy as np
import pandas as pd


def pdprint(*pandas_datas: pd.DataFrame) -> None:
    for data in pandas_datas:
        print(f"Data name: {data.name}, ", end="")
        print(f"Data Type: {data.dtypes}, ", end="")
        print(f"Data dim: {data.ndim}, ", end="")
        print(f"Data Shape: {data.shape}")
        print("-" * 20)
        print(data)