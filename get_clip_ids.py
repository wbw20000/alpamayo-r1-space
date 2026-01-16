"""Get all available clip_ids from PhysicalAI-Autonomous-Vehicles dataset"""
import os
from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

def get_all_clip_ids():
    """Download clip_index.parquet and extract all unique clip_ids"""
    print("Downloading clip_index.parquet from Official dataset...")

    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename="clip_index.parquet",
            repo_type="dataset",
        )
        print(f"Downloaded to: {local_path}")

        # Read parquet - clip_id is the INDEX
        df = pd.read_parquet(local_path)
        print(f"\nTotal clips: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # clip_id is the index
        clip_ids = df.index.tolist()

        # Get split distribution
        print(f"\nSplit distribution:")
        print(df['split'].value_counts())

        # Get chunk distribution
        print(f"\nChunk distribution (first 10):")
        print(df['chunk'].value_counts().head(10))

        # Save all clip_ids to file
        with open("clip_ids_all.txt", "w") as f:
            for cid in clip_ids:
                f.write(f"{cid}\n")
        print(f"\nSaved all {len(clip_ids)} clip_ids to clip_ids_all.txt")

        # Create a summary by split and chunk
        summary = df.reset_index().groupby(['split', 'chunk']).agg(
            count=('clip_id', 'count'),
            sample_clip=('clip_id', 'first')
        ).reset_index()

        print(f"\n{'=' * 80}")
        print("Sample clip_ids by split and chunk:")
        print("=" * 80)
        for _, row in summary.head(20).iterrows():
            print(f"  Split: {row['split']:5s} | Chunk: {row['chunk']:3d} | Count: {row['count']:5d} | Sample: {row['sample_clip']}")

        # Save summary
        summary.to_csv("clip_summary.csv", index=False)
        print(f"\nSaved summary to clip_summary.csv")

        return df

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_sample_clips_with_info(df, n=10):
    """Get sample clips with their info"""
    print(f"\n{'=' * 80}")
    print(f"Sample {n} clips for testing Official mode:")
    print("=" * 80)

    # Get samples from test split first, then train
    samples = []
    for split in ['test', 'train']:
        split_df = df[df['split'] == split]
        if len(split_df) > 0:
            sample_clips = split_df.head(n // 2 if split == 'test' else n - len(samples))
            for clip_id in sample_clips.index:
                row = split_df.loc[clip_id]
                samples.append({
                    'clip_id': clip_id,
                    'split': row['split'],
                    'chunk': row['chunk']
                })
                if len(samples) >= n:
                    break
        if len(samples) >= n:
            break

    print("\nClip ID                                | Split | Chunk")
    print("-" * 60)
    for s in samples:
        print(f"{s['clip_id']} | {s['split']:5s} | {s['chunk']:3d}")

    # Save sample clips
    with open("sample_clips.txt", "w") as f:
        f.write("# Sample clip_ids for Official mode testing\n")
        f.write("# Format: clip_id | split | chunk\n\n")
        for s in samples:
            f.write(f"{s['clip_id']}  # {s['split']}, chunk {s['chunk']}\n")
    print(f"\nSaved to sample_clips.txt")

    return samples

if __name__ == "__main__":
    df = get_all_clip_ids()

    if df is not None:
        samples = get_sample_clips_with_info(df, n=20)

        print("\n" + "=" * 80)
        print("Usage in Official mode:")
        print("=" * 80)
        print("""
1. Copy a clip_id from above
2. Paste it in the "Clip ID (UUID)" field
3. Set t0 (microseconds) - default 5100000 works for most clips
4. Click "Run Inference"

Note: The dataset contains camera videos at these paths:
  camera/{chunk}/{clip_id}.camera_front_wide_120fov.mp4
  camera/{chunk}/{clip_id}.camera_cross_left_120fov.mp4
  etc.

To preview a clip's video, you can use:
  https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles/resolve/main/camera/{chunk}/{clip_id}.camera_front_wide_120fov.mp4
""")
