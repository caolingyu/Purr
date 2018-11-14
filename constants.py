import os

LABEL_SIZE = 6

PAD_CHAR = "**PAD**"
EMBEDDING_SIZE = 200
MAX_LENGTH = 10000

# Where to save the model
MODEL_DIR = "./saved_models/treatment_type/"
DATA_DIR = "./data/"

for item in [MODEL_DIR, DATA_DIR]:
    if not os.path.exists(item):
        os.makedirs(item)
