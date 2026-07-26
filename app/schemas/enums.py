"""
enums.py

Enumerations for categorical features.
"""

from enum import Enum


class GarageQual(str, Enum):
    EX = "Ex"
    GD = "Gd"
    TA = "TA"
    FA = "Fa"
    PO = "Po"
    NA = "NA"


class GarageFinish(str, Enum):
    FIN = "Fin"
    RFN = "RFn"
    UNF = "Unf"
    NA = "NA"


class GarageType(str, Enum):
    ATTACHED = "Attchd"
    DETACHED = "Detchd"
    BUILTIN = "BuiltIn"
    BASEMENT = "Basment"
    CARPORT = "CarPort"
    TWOTYPES = "2Types"
    NA = "NA"


class KitchenQual(str, Enum):
    EX = "Ex"
    GD = "Gd"
    TA = "TA"
    FA = "Fa"
    PO = "Po"


class BsmtQual(str, Enum):
    EX = "Ex"
    GD = "Gd"
    TA = "TA"
    FA = "Fa"
    PO = "Po"
    NA = "NA"


class CentralAir(str, Enum):
    YES = "Y"
    NO = "N"


class MSZoning(str, Enum):
    RL = "RL"
    RM = "RM"
    FV = "FV"
    RH = "RH"
    C = "C"
    A = "A"


class LotShape(str, Enum):
    REG = "Reg"
    IR1 = "IR1"
    IR2 = "IR2"
    IR3 = "IR3"


class PavedDrive(str, Enum):
    YES = "Y"
    PARTIAL = "P"
    NO = "N"
