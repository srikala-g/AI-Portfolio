from datasets import load_dataset
import os

def load_incremental_data(save_dir, num_shards=10):
    print("🔹 Loading WikiArt dataset from Hugging Face...")
    dataset = load_dataset("huggan/wikiart", split="train", trust_remote_code=True)

    print(f"✅ Dataset loaded: {len(dataset)} samples")

    shard_size = len(dataset) // num_shards
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size if i < num_shards - 1 else len(dataset)
        shard = dataset.select(range(start, end))

        shard_path = os.path.join(save_dir, f"wikiart_train_shard_{i+1}.parquet")
        print(f"💾 Saving shard {i+1}/{num_shards} ({start}:{end}) → {shard_path}")
        shard.to_parquet(shard_path)

    print("🎉 All shards saved successfully!")

if __name__ == "__main__":
    load_incremental_data(
        save_dir="/Users/srikala/projects/AI-Portfolio/art_identifier/data/dataset/wikiart/data",
        num_shards=5,  # adjust depending on how much you want to process
    )
