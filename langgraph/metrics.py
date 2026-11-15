import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

basepath = Path(__file__).parent

df = pd.read_csv(basepath / "orchestrated_results_updated.csv")
print(classification_report(df["true_label"], df["predicted_grade"], digits=4))
print(confusion_matrix(df["true_label"], df["predicted_grade"]))