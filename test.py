import pickle
import time


def load_and_compare(target_number):
    # Load pregenerated numbers
    with open('saved_numbers.pkl', 'rb') as f:
        numbers = pickle.load(f)

    # Perform comparison
    start_time = time.perf_counter()
    matches = numbers.count(target_number)
    elapsed = time.perf_counter() - start_time

    print(f"Found {matches} matches for {target_number}")
    print(f"Comparison time: {elapsed:.4f} seconds")


if __name__ == "__main__":
    target = int(input("Enter number to compare (0-1000): "))
    load_and_compare(target)