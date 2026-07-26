"""
Default values for features not entered by the user.

These defaults are merged with the user input before prediction.
"""

DEFAULT_FEATURES = {
    # ---------- Lot ----------
    "LotFrontage": 70.0,
    "Street": "Pave",
    "Alley": "NA",
    "LotConfig": "Inside",
    "LandContour": "Lvl",
    "LandSlope": "Gtl",
    "Utilities": "AllPub",
    # ---------- Exterior ----------
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0.0,
    "ExterCond": "TA",
    # ---------- Basement ----------
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Unf",
    "BsmtFinSF1": 0,
    "BsmtFinType2": "Unf",
    "BsmtFinSF2": 0,
    "BsmtUnfSF": 0,
    # ---------- Heating ----------
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "Electrical": "SBrkr",
    # ---------- Garage ----------
    "GarageYrBlt": 2000,
    "GarageCond": "TA",
    # ---------- Porch ----------
    "WoodDeckSF": 0,
    "OpenPorchSF": 0,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    # ---------- Pool ----------
    "PoolArea": 0,
    "PoolQC": "NA",
    # ---------- Fence ----------
    "Fence": "NA",
    # ---------- Misc ----------
    "MiscFeature": "NA",
    "MiscVal": 0,
    # ---------- Sale ----------
    "MoSold": 6,
    "SaleType": "WD",
    "SaleCondition": "Normal",
}
