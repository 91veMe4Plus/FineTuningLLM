import os
from glob import glob
import pandas as pd


def load_and_combine_data(data_dir: str) -> pd.DataFrame:
    """Load and combine all data from CSV files in the specified directory."""
    data_files = glob(os.path.join(data_dir, "*_난독화결과.csv"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}. Please check the path.")

    all_data = []
    print("Loading data files...")
    for file_path in data_files:
        try:
            df = pd.read_csv(file_path)
            # Extract category from filename, e.g., "구어체_대화체_16878_sample_난독화결과.csv" -> "구어체"
            category = os.path.basename(file_path).split('_')[0]
            df['category'] = category
            all_data.append(df)
            print(f"  - Loaded {os.path.basename(file_path)}: {len(df)} samples")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    if not all_data:
        raise ValueError("No data could be loaded. Please check the CSV files.")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal dataset size: {len(combined_df)}")
    print("Category distribution:")
    print(combined_df['category'].value_counts())
    return combined_df


def create_instruction_dataset(df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    """Create an instruction-formatted dataset for the deobfuscation task."""
    if sample_size:
        print(f"\nSampling {sample_size} instances from the dataset...")
        # Ensure sample size is not larger than the dataframe
        n_samples = min(sample_size, len(df))
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    instructions = []
    for _, row in df.iterrows():
        instruction = {
            'input': f"다음 난독화된 한국어 텍스트를 원래 텍스트로 복원해주세요.\n\n난독화된 텍스트: {row['obfuscated']}",
            'output': row['original'],
            'category': row['category']
        }
        instructions.append(instruction)

    return pd.DataFrame(instructions) 