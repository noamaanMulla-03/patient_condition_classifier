from src.data_loader import load_data

def main():
    dataset = load_data()
    print(f"Loaded train and test datasets ({dataset['train']}) ({dataset['test']})")


if __name__ == "__main__":
    main()
