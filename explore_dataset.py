"""Explore PhysicalAI-Autonomous-Vehicles dataset structure"""
import os
from huggingface_hub import hf_hub_download, list_repo_tree
import pandas as pd

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

def explore_repo_structure():
    """List all files in the dataset repository"""
    print("Exploring repository structure...")
    print("=" * 60)

    try:
        # List top-level files
        files = list(list_repo_tree(REPO_ID, repo_type="dataset", recursive=False))
        print(f"Found {len(files)} items at root level:")
        for f in files:
            print(f"  {f.path} ({f.type})")

        # Check subdirectories
        for subdir in ['calibration', 'camera', 'labels', 'metadata']:
            print(f"\n--- {subdir}/ ---")
            try:
                sub_files = list(list_repo_tree(REPO_ID, path_in_repo=subdir, repo_type="dataset", recursive=False))
                for sf in sub_files[:10]:  # First 10 files
                    print(f"  {sf.path}")
                if len(sub_files) > 10:
                    print(f"  ... and {len(sub_files) - 10} more files")
            except Exception as e:
                print(f"  Error: {e}")

    except Exception as e:
        print(f"Error: {e}")

def download_and_inspect(filename):
    """Download a file and inspect its contents"""
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {filename}")
    print("=" * 60)

    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
        )
        print(f"Downloaded to: {local_path}")

        if filename.endswith('.parquet'):
            df = pd.read_parquet(local_path)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            print(f"\nData types:")
            print(df.dtypes)

            # Check for clip-related columns
            for col in df.columns:
                if 'clip' in col.lower() or 'id' in col.lower():
                    unique_vals = df[col].nunique()
                    print(f"\nColumn '{col}' has {unique_vals} unique values")
                    if unique_vals < 20:
                        print(f"  Values: {df[col].unique().tolist()}")

        elif filename.endswith('.csv'):
            df = pd.read_csv(local_path)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 10 rows:")
            print(df.head(10))

        return local_path

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # First explore structure
    explore_repo_structure()

    # Inspect key files
    download_and_inspect("features.csv")
    download_and_inspect("clip_index.parquet")

    # Try to find clip info in metadata
    print("\n" + "=" * 60)
    print("Looking for clip information...")
    print("=" * 60)
