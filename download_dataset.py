import os
os.environ["KAGGLEHUB_CACHE"] = "D:/kagglehub"  # Windows example

import kagglehub
path = kagglehub.dataset_download("mohlamin/resized-eyepacs-diabetic-retinopathy-dataset")
print("Path to dataset files:", path)
