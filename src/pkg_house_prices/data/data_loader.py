from pathlib import Path
import pandas as pd
from pkg_house_prices.utils.config import CONFIG

def load_data(CONFIG):

    train_path = Path(CONFIG["data"]["train"])
    test_path  = Path(CONFIG["data"]["test"])

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test
