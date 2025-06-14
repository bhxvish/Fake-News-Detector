import pandas as pd

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index = True)
df = df[["text","label"]]
df.dropna(inplace=True)

df.to_csv("train.csv",index = False)

