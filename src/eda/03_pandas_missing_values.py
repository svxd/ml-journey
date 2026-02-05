import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age": [25, None, None, 30],
    "salary": [50000, 60000, None, 70000],
    "department": ["IT", "HR", "IT", None]
})


# A
# print(df.isna().sum())

# B
# df["age"] = df["age"].fillna(df["age"].mean())
# df["salary"] = df["salary"].fillna(df["salary"].median())
# print(df)

# C
df = df.dropna(subset=(["department"]))
print(df)

# D
# print(df["department"].value_counts())



















# print
# print(df)
# print(df.isna())
# print(df.isna().sum())

# print(df["salary"].fillna(df["salary"].mean()))
# print(df["age"].fillna(df["age"].mean()))
# print(df.dropna())