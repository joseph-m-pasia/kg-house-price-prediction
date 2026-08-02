"""
Default values for features not entered by the user.

These defaults are merged with the user input before prediction.
"""

DEFAULT_FEATURES = {
    # ---------- Identification ----------
    "Id": 0,
    # ---------- Building ----------
    "MSSubClass": 20,
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    # ---------- Lot ----------
    "LotFrontage": 70.0,
    "LotArea": 10000,
    "Street": "Pave",
    "Alley": "NA",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "Condition2": "Norm",
    # ---------- Exterior ----------
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "MasVnrArea": 0.0,
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "PConc",
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
    # ---------- Rooms ----------
    "1stFlrSF": 1000,
    "2ndFlrSF": 0,
    "LowQualFinSF": 0,
    "BedroomAbvGr": 3,
    "TotRmsAbvGrd": 6,
    "Functional": "Typ",
    "FireplaceQu": "NA",
    # ---------- Garage ----------
    "GarageYrBlt": 2000,
    "GarageArea": 0,
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
    # ---------- Engineered Features ----------
    "TotalSF": 1000,
    "HasPool": 0,
    "HasSecondFloor": 0,
    "HasDeck": 0,
    "HasFence": 0,
}
