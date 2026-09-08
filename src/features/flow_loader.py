import pandas as pd


def inspect_dataset(path, chunk_size=100000):

    total_rows = 0
    label_counts = {}

    for chunk in pd.read_csv(path, chunksize=chunk_size):

        total_rows += len(chunk)

        counts = chunk["Label"].value_counts()

        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + count

    print("Total rows:", total_rows)

    print("\nAll labels:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {count}")


if __name__ == "__main__":

    path = "data/raw/CTU-13/capture20110810.binetflow"

    inspect_dataset(path)