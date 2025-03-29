import requests
from time import sleep
import random
import nltk
from nltk.corpus import wordnet as wn



host = ""
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"

NUM_ROUNDS = 5

# Word counter relationships (define more as needed)
counter_map = {
    "Feather": ["coal", "rock", "water", "earthquake", "flame", "wind", "ice", "fire", "paper", "light"],
    "Coal": ["water", "fire", "ice", "earthquake", "volcano", "flame", "rock", "plague", "wind", "sandstorm"],
    "Pebble": ["rock", "coal", "stone", "sandstorm", "water", "paper", "disease", "storm", "plague", "light"],
    "Leaf": ["fire", "water", "paper", "rock", "light", "sandstorm", "plague", "wind", "earthquake", "stone"],
    "Paper": ["water", "fire", "wind", "light", "earth", "plague", "stone", "sandstorm", "coal", "leaf"],
    "Rock": ["feather", "coal", "water", "paper", "flame", "earth", "light", "disease", "sandstorm", "ice"],
    "Water": ["fire", "earth", "rock", "disease", "ice", "wind", "plague", "coal", "storm", "light"],
    "Twig": ["fire", "earthquake", "wind", "light", "water", "plague", "stone", "leaf", "sandstorm", "paper"],
    "Sword": ["shield", "war", "wind", "logic", "light", "storm", "earth", "magma", "peace", "fire"],
    "Shield": ["fire", "wind", "rock", "earthquake", "water", "disease", "peace", "logic", "time", "sandstorm"],
    "Gun": ["shield", "earthquake", "fire", "logic", "plague", "water", "wind", "stone", "light", "ice"],
    "Flame": ["water", "shield", "fire", "wind", "earthquake", "plague", "ice", "sandstorm", "stone", "disease"],
    "Rope": ["earthquake", "plague", "water", "wind", "rock", "disease", "magma", "light", "fire", "storm"],
    "Disease": ["water", "cure", "plague", "rebirth", "shield", "light", "storm", "magma", "fate", "time"],
    "Cure": ["disease", "water", "flame", "rebirth", "earth", "logic", "peace", "light", "wind", "magma"],
    "Bacteria": ["cure", "disease", "flame", "plague", "rebirth", "water", "earth", "storm", "light", "fate"],
    "Shadow": ["light", "sun", "moon", "storm", "plague", "earth", "star", "time", "wind", "supernova"],
    "Light": ["shadow", "star", "supernova", "plague", "storm", "earth", "fate", "water", "ice", "time"],
    "Virus": ["water", "plague", "cure", "rebirth", "earthquake", "flame", "wind", "disease", "stone", "magma"],
    "Sound": ["water", "earth", "fire", "magma", "plague", "wind", "storm", "time", "stone", "light"],
    "Time": ["logic", "fate", "peace", "earth", "wind", "rebirth", "sound", "sun", "star", "light"],
    "Fate": ["time", "peace", "light", "storm", "rebirth", "earth", "wind", "logic", "sound", "supernova"],
    "Earthquake": ["water", "wind", "earth", "storm", "light", "sandstorm", "fire", "magma", "plague", "fate"],
    "Storm": ["earth", "wind", "water", "light", "ice", "magma", "sun", "fire", "plague", "sandstorm"],
    "Vaccine": ["disease", "rebirth", "water", "plague", "fire", "storm", "cure", "earth", "flame", "magma"],
    "Logic": ["time", "peace", "wind", "rebirth", "earthquake", "sound", "flame", "cure", "storm", "sandstorm"],
    "Gravity": ["wind", "earth", "time", "peace", "flame", "logic", "supernova", "sun", "ice", "water"],
    "Robots": ["time", "earth", "stone", "flame", "fire", "storm", "sandstorm", "cure", "rebirth", "magma"],
    "Stone": ["rock", "flame", "wind", "fire", "water", "earth", "cure", "time", "sun", "storm"],
    "Echo": ["sound", "time", "earth", "flame", "light", "wind", "water", "sun", "ice", "storm"],
    "Thunder": ["wind", "water", "storm", "time", "supernova", "ice", "light", "plague", "earth", "sound"],
    "Karma": ["war", "time", "light", "earth", "peace", "supernova", "rebirth", "wind", "sandstorm", "storm"],
    "Wind": ["fire", "sandstorm", "stone", "time", "earth", "water", "light", "magma", "storm", "disease"],
    "Ice": ["fire", "water", "earthquake", "sandstorm", "flame", "wind", "plague", "stone", "sun", "light"],
    "Sandstorm": ["wind", "water", "earth", "fire", "ice", "flame", "storm", "plague", "sand", "time"],
    "Laser": ["fire", "wind", "earthquake", "magma", "stone", "flame", "rock", "sun", "water", "plague"],
    "Magma": ["wind", "stone", "water", "earth", "ice", "fire", "sandstorm", "plague", "flame", "robots"],
    "Peace": ["war", "logic", "karma", "earth", "rebirth", "rebirth", "time", "sound", "flame", "storm"],
    "Explosion": ["sandstorm", "earthquake", "water", "flame", "wind", "earth", "stone", "supernova", "plague", "fire"],
    "War": ["peace", "rebirth", "explosion", "karma", "wind", "earth", "fire", "flame", "magma", "storm"],
    "Enlightenment": ["war", "rebirth", "karma", "earth", "peace", "time", "logic", "stone", "wind", "plague"],
    "NuclearBomb": ["earthquake", "plague", "storm", "wind", "water", "sun", "stone", "plague", "explosion", "magma"],
    "Volcano": ["earthquake", "magma", "water", "wind", "sandstorm", "storm", "fire", "plague", "ice", "sun"],
    "Whale": ["storm", "water", "earthquake", "plague", "ice", "flood", "drought", "earth", "sun", "wind"],
    "Earth": ["stone", "fire", "water", "light", "time", "logic", "wind", "sandstorm", "earthquake", "ice"],
    "Moon": ["earth", "sun", "time", "light", "storm", "space", "wind", "star", "gravity", "black hole"],
    "Star": ["moon", "light", "sun", "earth", "wind", "storm", "time", "gravity", "space", "plague"],
    "Tsunami": ["flood", "earthquake", "fire", "wind", "stone", "magma", "water", "plague", "sandstorm", "sun"],
    "Supernova": ["black hole", "star", "earth", "time", "light", "space", "plague", "storm", "wind", "gravity"],
    "Antimatter": ["earth", "black hole", "space", "sun", "supernova", "gravity", "plague", "moon", "star", "tsunami"],
    "Plague": ["rebirth", "cure", "wind", "fire", "stone", "disease", "flood", "water", "magma", "earth"],
    "Rebirth": ["plague", "cure", "wind", "time", "stone", "sun", "war", "peace", "storm", "water"],
    "TectonicShift": ["earthquake", "sandstorm", "water", "fire", "stone", "sun", "magma", "light", "plague", "storm"],
    "GammaRayBurst": ["plague", "earth", "storm", "fire", "tsunami", "wind", "water", "light", "sun", "supernova"],
    "HumanSpirit": ["war", "peace", "time", "stone", "light", "logic", "rebirth", "magma", "sandstorm", "earth"],
    "ApocalypticMeteor": ["nuclear bomb", "storm", "earthquake", "plague", "sun", "water", "sandstorm", "tsunami", "storm", "wind"],
    "EarthsCore": ["volcano", "earthquake", "magma", "storm", "plague", "sandstorm", "water", "tsunami", "flood", "space"],
    "NeutronStar": ["gravity", "space", "supernova", "black hole", "time", "star", "plague", "storm", "tsunami", "flood"],
    "SupermassiveBlackHole": ["space", "gravity", "supernova", "plague", "storm", "time", "sun", "earth", "light", "tsunami"],
    "Entropy": ["energy", "universe", "space", "supernova", "time", "gravity", "plague", "black hole", "sun", "tsunami"]
}

# Word costs (subset shown, complete this list)
word_costs = {
    "water": 3, "sandstorm": 13, "extinguisher": 10,
    "fire": 5, "magma": 14, "warmth": 7,
    "shield": 4, "earth": 17, "time": 8,
    "peace": 14, "logic": 10, "enlightenment": 15,
    "vaccine": 9, "cure": 6, "rebirth": 20,
    "light": 7, "star": 18, "supernova": 19,
    "drought": 7, "wind": 13
}


def get_definition(word):
    """Retrieve definition words from NLTK's WordNet."""
    synsets = wn.synsets(word)
    if synsets:
        definition = synsets[0].definition()
        return set(definition.lower().split())  # Simple split, can be improved
    return set()


def what_beats(system_word):
    """Find the cheapest countering word based on the system word definition."""
    definition_words = get_definition(system_word)
    best_choice = None
    best_cost = float('inf')

    for key, counters in counter_map.items():
        if key in definition_words:
            for counter in counters:
                if counter in word_costs and word_costs[counter] < best_cost:
                    best_choice = counter
                    best_cost = word_costs[counter]

    if best_choice:
        print(f"System word: {system_word} → Choosing {best_choice} (${best_cost})")
        return best_choice  # Return the word instead of ID

    # Fallback: pick the cheapest general word
    return min(word_costs, key=word_costs.get)


print(get_definition("light"))

# def play_game(player_id):
#     for round_id in range(1, NUM_ROUNDS + 1):
#         round_num = -1
#         while round_num != round_id:
#             response = requests.get(get_url)
#             sys_word = response.json()['word']
#             round_num = response.json()['round']
#             sleep(1)
#
#         if round_id > 1:
#             status = requests.get(status_url)
#             print(status.json())
#
#         chosen_word = what_beats(sys_word)
#         word_id = list(word_costs.keys()).index(chosen_word) + 1  # Convert word to ID
#         data = {"player_id": player_id, "word_id": word_id, "round_id": round_id}
#         response = requests.post(post_url, json=data)
#         print(response.json())
