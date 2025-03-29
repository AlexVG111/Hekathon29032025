import random
import pickle

NUM_ELEMENTS = 10_000_000  # 10 million numbers
RANGE_START = 0
RANGE_END = 1000


def generate_and_save():
    print(f"Generating {NUM_ELEMENTS:,} random numbers...")
    numbers = [random.randint(RANGE_START, RANGE_END) for _ in range(NUM_ELEMENTS)]

    with open('saved_numbers.pkl', 'wb') as f:
        pickle.dump(numbers, f)
    print(f"Saved to saved_numbers.pkl")


if __name__ == "__main__":
    generate_and_save()