import pandas as pd

data = {
    "age": [22, 35, 58, 41],
    "salary": [40000, 80000, 150000, 90000],
    "experience": [1, 10, 30, 15]
}

df = pd.DataFrame(data)
# print(df)


# print(df.head(2))
# print(f"{'---------' * 3}")
# print(df.shape)
# print(f"{'---------' * 3}")
# df.info()
# print(f"{'---------' * 3}")
# print(df.mean())
# print(f"{'---------' * 3}")
print(df.loc[1, "salary"])


# print(df[df["salary"] > 80000]) 
# print(df[(df["age"] > 35) & (df["experience"] >= 10)])

# print(df.sort_values("salary", ascending=False))


