import os
os.environ["KAGGLEHUB_CACHE"] = "D:/kagglehub"  # Windows example

import kagglehub
# path = kagglehub.dataset_download("mohlamin/resized-eyepacs-diabetic-retinopathy-dataset")
path = kagglehub.dataset_download('aaryapatel98/indian-diabetic-retinopathy-image-dataset')
print("Path to dataset files:", path)
