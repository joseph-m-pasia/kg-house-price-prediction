"""
schemas.py

Input and output schemas for the House Price Predictor.
"""

from pydantic import BaseModel, ConfigDict, Field

# ============================================================
# ENUMS
# ============================================================

from app.schemas.enums import (
    BsmtQual,
    CentralAir,
    GarageFinish,
    GarageQual,
    GarageType,
    KitchenQual,
    LotShape,
    MSZoning,
    PavedDrive,
)

# ============================================================
# USER INPUT
# ============================================================


class PredictionRequest(BaseModel):

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # ----------------------------
    # Overall Quality
    # ----------------------------

    OverallQual: int = Field(ge=1, le=10)
    OverallCond: int = Field(ge=1, le=10)

    # ----------------------------
    # Living Area
    # ----------------------------

    TotalBsmtSF: float = Field(ge=0)
    FirstFlrSF: float = Field(alias="1stFlrSF", ge=0)
    SecondFlrSF: float = Field(alias="2ndFlrSF", ge=0)

    GrLivArea: float = Field(gt=0)

    # ----------------------------
    # Bathrooms
    # ----------------------------

    FullBath: int = Field(ge=0)
    HalfBath: int = Field(ge=0)
    BsmtFullBath: int = Field(ge=0)
    BsmtHalfBath: int = Field(ge=0)

    # ----------------------------
    # Kitchen
    # ----------------------------

    KitchenQual: KitchenQual
    KitchenAbvGr: int = Field(ge=0)

    # ----------------------------
    # Garage
    # ----------------------------

    GarageCars: int = Field(ge=0)

    GarageQual: GarageQual

    GarageFinish: GarageFinish

    GarageType: GarageType

    # ----------------------------
    # Basement
    # ----------------------------

    BsmtQual: BsmtQual

    # ----------------------------
    # Fireplaces
    # ----------------------------

    Fireplaces: int = Field(ge=0)

    # ----------------------------
    # Cooling
    # ----------------------------

    CentralAir: CentralAir

    # ----------------------------
    # Lot
    # ----------------------------

    LotShape: LotShape

    # ----------------------------
    # Zoning
    # ----------------------------

    MSZoning: MSZoning

    # ----------------------------
    # Driveway
    # ----------------------------

    PavedDrive: PavedDrive

    # ----------------------------
    # Dates
    # ----------------------------

    YearBuilt: int = Field(ge=1800, le=2025)

    YearRemodAdd: int = Field(ge=1800, le=2025)

    YrSold: int = Field(ge=2006, le=2025)


# ============================================================
# OUTPUT
# ============================================================


class PredictionResponse(BaseModel):

    predicted_sale_price: float
