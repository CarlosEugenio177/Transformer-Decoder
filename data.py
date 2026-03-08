#Houve utilização de Inteligência Artificial para revisão e adequação do codigo aos requisitos estabelecidos.
import pandas as pd
import numpy as np

Vocabulary = {"No": 0, "i": 1, "am": 2, "your": 3, "father": 4, ".": 5}

tokens = ["No", "i", "am", "your", "father", "."]
ids = [0, 1, 2, 3, 4, 5]

dfvocab = pd.DataFrame(list(Vocabulary.items()),columns=["word", "id"])

print(dfvocab)
print("ids: "+str(ids))

sizevocab = len(Vocabulary)

d_model = 64

embedding_matrix = np.random.rand(sizevocab, d_model)
print("embedding_matrix shape: "+str(embedding_matrix.shape))

sentenceembeddings = [embedding_matrix[id] for id in ids]

print(sentenceembeddings)

# criação do tensor de entrada para o encoder
X = np.expand_dims(sentenceembeddings, axis=0)

print("X shape: "+str(X.shape))