from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
OUTPUT_DIR = ROOT / "outputs"

TARGET = "Natural_Gas_US_Henry_Hub_Gas"
HORIZONS = (1, 2, 3)
TRAIN_END = pd.Timestamp("2020-08-31")

# until 2012 the target is the spot price, from 2013 it is the front-month futures price
TARGET_DEFINITION_CHANGE = pd.Timestamp("2013-01-31")

# storage, production, consumption etc. are published ~2 months late
PUBLICATION_LAG_MONTHS = 2

# for choosing the model: train up to 2012, test on 2013 - Aug 2020
PRACTICE_TRAIN_END = pd.Timestamp("2012-12-31")
