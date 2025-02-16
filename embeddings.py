import os
import gensim.downloader as api
import gensim
EMBEDDING_MODEL_PATH = "glove_model.model"

if os.path.exists(EMBEDDING_MODEL_PATH):
    print("Load saved model embeddings...")
    embeddings = gensim.models.KeyedVectors.load(EMBEDDING_MODEL_PATH)
else:
    print("Download and save GloVe embeddings...")
    embeddings = api.load("glove-wiki-gigaword-300")
    embeddings.save(EMBEDDING_MODEL_PATH)

print("Embeddings have been saved to file", EMBEDDING_MODEL_PATH)