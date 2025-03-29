import json
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression

def load_glove(path):
    return {line.split()[0]: np.array(line.split()[1:], dtype='float32')
            for line in open(path, encoding='utf-8')}

# Step 1: Load categories
with open('categories.json') as f:
    categories = json.load(f)

# Step 2: Load and filter GloVe
embeddings = load_glove('glove.6B.50d.txt')
filtered_embeddings = {
    word: embeddings[word]
    for word in sum(categories.values(), [])
    if word in embeddings
}
dump(filtered_embeddings, 'filtered_embeddings.joblib')

# Step 3: Train classifier
X_train = [filtered_embeddings[word] for word in sum(categories.values(), [])]
y_train = [cat for cat, words in categories.items() for _ in words]
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
dump(model, 'classifier.joblib')

print("Preprocessing complete!")