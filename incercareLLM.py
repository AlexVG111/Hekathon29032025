import spacy
import time
import numpy
# Load the spaCy model
nlp = spacy.load("en_core_web_md")

# Define categories and associated keywords
categories = {
    "water": ["ocean", "river", "sea", "lake", "water", "rain", "wave"],
    "fire": ["fire", "flame", "heat", "burn", "inferno", "blaze"],
    "physical force": ["force", "pressure", "strength", "tension", "push", "impact"],
    "calm and control": ["calm", "peace", "tranquility", "control", "balance", "stillness"],
    "time and fate": ["time", "destiny", "fate", "clock", "future", "past", "moment"],
    "disease": ["disease", "illness", "virus", "bacteria", "infection", "sickness"],
    "natural elements": ["earth", "wind", "fire", "water", "sky", "stone", "soil"],
    "abstract forces": ["emotion", "thought", "mind", "idea", "energy", "willpower"],
    "technology": ["technology", "machine", "robot", "computer", "device", "electronics"],
    "space": ["space", "universe", "galaxy", "star", "planet", "black hole"],
    "impact": ["impact", "collision", "crash", "smash", "strike", "hit"],
    "cosmic force": ["gravity", "dark matter", "big bang", "cosmos", "black hole", "universe"]
}


# Function to categorize a word
def categorize_word(word):
    doc = nlp(word)
    word_vector = doc.vector

    best_category = None
    highest_similarity = -1

    # Loop through categories and compare word vectors
    for category, keywords in categories.items():
        category_vector = sum(nlp(keyword).vector for keyword in keywords) / len(keywords)
        similarity = word_vector.dot(category_vector) / (doc.vector_norm * category_vector.norm)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_category = category

    return best_category


# Example usage
wordtofind = "fire"
start_time = time.time()
category = categorize_word(wordtofind)
end_time = time.time()

print(f"The word '{wordtofind}' belongs to the category: {category}")
print(f"Time taken: {end_time - start_time:.3f} seconds")
