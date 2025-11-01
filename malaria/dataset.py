import tensorflow_datasets as tfds

# Download + prepare the dataset locally
builder = tfds.builder("malaria")
builder.download_and_prepare()

# This path contains the raw extracted dataset including `cell_images`
data_dir = builder.data_dir
print("Data directory:", data_dir)
