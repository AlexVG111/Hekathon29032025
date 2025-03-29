import numpy as np
from numpy.linalg import norm
import time
import requests

# Start full execution timer
full_start = time.perf_counter()


def load_glove(embedding_path):
    """Load GloVe embeddings with progress tracking"""
    embeddings = {}
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            embeddings[parts[0]] = np.array(parts[1:], dtype='float32')
    return embeddings


def quick_check(embeddings, target, word, threshold=0.5):
    """Ultra-fast YES/NO compatibility check"""
    if target not in embeddings or word not in embeddings:
        return "NO"

    tv, wv = embeddings[target], embeddings[word]
    dot = np.dot(tv, wv)
    if dot < 0:
        return "NO"

    min_mag_sq = min(norm(tv) ** 2, norm(wv) ** 2)
    return "YES" if dot >= threshold * min_mag_sq else "NO"


# Load embeddings
print("Loading word vectors...")
load_start = time.perf_counter()
glove = load_glove('glove.6B.100d.txt')
load_time = time.perf_counter() - load_start
print(f"Loaded in {load_time:.1f}sec\n")

# Define test cases
target_word = "fossil"
comparison_words = [
    # Musical elements (30)
    "fragile"
]

# Execute all comparisons
print(f"Testing '{target_word}' against {len(comparison_words)} words:")
results = []
check_start = time.perf_counter()

for word in comparison_words:
    results.append((word, quick_check(glove, target_word, word)))

check_time = time.perf_counter() - check_start
total_time = time.perf_counter() - full_start

# Display summary results
print("\nCategory Breakdown:")
categories = {
    "Musical Elements": comparison_words[:30],
    "Genres": comparison_words[30:50],
    "Related Concepts": comparison_words[50:70],
    "Unrelated Terms": comparison_words[70:90],
    "Opposites": comparison_words[90:]
}

for cat, words in categories.items():
    yes_count = sum(1 for w in words if quick_check(glove, target_word, w) == "YES")
    print(f"{cat:<18}: {yes_count}/{len(words)} matches")

# Formatted timing
print(f"\nTiming:")
print(f"- Embeddings loaded: {load_time:.1f}sec")
print(f"- 100 comparisons: {check_time:.3f}sec")
print(f"- Total execution: {total_time:.1f}sec")
print(f"- Avg per check: {check_time / 100:.6f}sec")


#
# NUM_ROUNDS = 5
#
# host = "http://172.18.4.158:8000"
# post_url = f"{host}/submit-word"
# get_url = f"{host}/get-word"
# status_url = f"{host}/status"
#
# for round_id in range(1, NUM_ROUNDS+1):
#         round_num = -1
#         while round_num != round_id:
#             response = requests.get(get_url)
#             print(response.json())
#             sys_word = response.json()['word']
#             round_num = response.json()['round']
#
#             time.sleep(1)
#
#         if round_id > 1:
#             status = requests.get(status_url)
#             print(status.json())
#
#         choosen_word = what_beats(sys_word)
#         data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_id}
#         response = requests.post(post_url, json=data)
#         print(response.json())