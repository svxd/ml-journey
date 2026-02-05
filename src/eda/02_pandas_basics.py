import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eva"],
    "age": [25, 40, 35, 29, 50],
    "experience": [2, 15, 10, 5, 20],
    "salary": [50000, 90000, 80000, 60000, 120000],
    "department": ["IT", "HR", "IT", "Marketing", "IT"]
})

# # print(df)
# print(df["department"].value_counts())



# A
# df["salary_k"] = df["salary"] / 1000

# B
# print(df[df["department"] == "IT"])

# C
# df["senior"] =  df["experience"] >= 10

# D
print(df.groupby("department")["salary"].mean().round())

# print(df.head(2))
# print(f"{'---------' * 3}")
# print(df.shape)
# print(f"{'---------' * 3}")
# df.info()
# print(f"{'---------' * 3}")
# print(df.mean())
# print(f"{'---------' * 3}")
# print(df.loc[1, "salary"])


# print(df[df["salary"] > 80000]) 
# print(df[(df["age"] > 35) & (df["experience"] >= 10)])

# print(df.sort_values("salary", ascending=False))


