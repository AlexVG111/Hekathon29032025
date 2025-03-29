import numpy as np
from numpy.linalg import norm
import time

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
glove = load_glove('glove.6B.50d.txt')
load_time = time.perf_counter() - load_start
print(f"Loaded in {load_time:.1f}sec\n")

# Define test cases
target_word = "cat"
comparison_words = [
    # Musical elements (30)
    "song", "melody", "instrument", "piano", "guitar", "violin", "drum", "flute",
    "harp", "trumpet", "saxophone", "cello", "clarinet", "banjo", "harmonica",
    "concert", "jazz", "orchestra", "rhythm", "harmony", "beat", "tempo", "chord",
    "note", "scale", "lyrics", "verse", "chorus", "album", "playlist",

    # Genres (20)
    "rock", "pop", "classical", "blues", "reggae", "hiphop", "electronic", "folk",
    "country", "metal", "punk", "disco", "soul", "funk", "latin", "indie",
    "alternative", "rap", "techno", "house",

    # Related concepts (20)
    "dance", "artist", "creative", "performance", "studio", "sound", "audio", "record",
    "composer", "singer", "band", "musician", "concert", "festival", "talent",
    "rehearsal", "microphone", "speaker", "headphones", "spotify",

    # Unrelated terms (20)
    "car", "computer", "tree", "ocean", "mathematics", "basketball", "coffee",
    "mountain", "airplane", "book", "painting", "science", "history", "phone",
    "garden", "kitchen", "shoes", "weather", "newspaper", "pillow",

    # Opposites/contrasts (10)
    "silence", "noise", "static", "quiet", "stillness", "mute", "hush",
    "calm", "peace", "carnivore"
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

# Optional: Save full results to file
with open("music_comparisons.txt", "w") as f:
    f.write(f"Results for '{target_word}' vs 100 words:\n")
    for word, res in results:
        f.write(f"{word:<15}: {res}\n")
    f.write(f"\nCompleted in {total_time:.1f} seconds")