# tests/test_load_data.py
from pathlib import Path
import sys
import yaml

# Add src folder to sys.path so Python can find your package
sys.path.append(str(Path(__file__).parents[2] / "src"))

from pkg_house_prices.data.data_loader import load_data

# Load the config
config_path = Path(__file__).parents[2] / "config.yaml"
with open(config_path, "r") as f:
    CONFIG = yaml.safe_load(f)

# Test the function
if __name__ == "__main__":
    train_df, test_df = load_data(CONFIG)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Columns:", train_df.columns.tolist())
