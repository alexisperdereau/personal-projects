import kagglehub

# Download latest version
path = kagglehub.dataset_download("nitindatta/finance-data")

print("Path to dataset files:", path)