import pandas as pd
from pathlib import Path
from pkg_house_prices.utils.config import CONFIG  # import the shared CONFIG


def load_data(train_path, test_path):

    train_path = Path(CONFIG["data"]["train"])
    test_path  = Path(CONFIG["data"]["test"])

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test