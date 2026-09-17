import pandas as pd
df = pd.read_csv(r"D:\Thesis Dataset\labels\sharpness_manifest.csv")
df = df[df["image_exists"]]
print(df["level"].value_counts().sort_index())