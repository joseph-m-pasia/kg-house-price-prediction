console.log("My script.js loaded");

const form = document.getElementById("prediction-form" );
const resultSection = document.getElementById("prediction-result");
const predictedPrice = document.getElementById("predicted-price");
const errorMessage = document.getElementById("error-message");
const predictButton = document.getElementById("predict-button");

console.log("form =", form);
console.log("button =", predictButton);

// =========================================================
// FORM SUBMISSION
// =========================================================


console.log("ABOUT TO REGISTER SUBMIT HANDLER");


form.addEventListener("click", async function (event) {

    console.log("SUBMIT EVENT FIRED");


    // Clear previous messages
    resultSection.hidden = true;
    errorMessage.hidden = true;

    // Disable button while prediction is running
    predictButton.disabled = true;
    predictButton.textContent = "Calculating...";


// =====================================================
// COLLECT FORM VALUES
// =====================================================

    const data = {

    // -----------------------------
    // Property Quality
    // -----------------------------

        OverallQual: Number(
            document.getElementById("overall-qual").value
        ),

        OverallCond: Number(
            document.getElementById("overall-cond").value
        ),


    // -----------------------------
    // Living Area
    // -----------------------------

        TotalBsmtSF: Number(
            document.getElementById("total-bsmt-sf").value
        ),

        "1stFlrSF": Number(
            document.getElementById("first-flr-sf").value
        ),

        "2ndFlrSF": Number(
            document.getElementById("second-flr-sf").value
        ),

        GrLivArea: Number(
            document.getElementById("gr-liv-area").value
        ),


    // -----------------------------
    // Bathrooms
    // -----------------------------

        FullBath: Number(
            document.getElementById("full-bath").value
        ),

        HalfBath: Number(
            document.getElementById("half-bath").value
        ),

        BsmtFullBath: Number(
            document.getElementById("bsmt-full-bath").value
        ),

        BsmtHalfBath: Number(
            document.getElementById("bsmt-half-bath").value
        ),


    // -----------------------------
    // Kitchen
    // -----------------------------

        KitchenQual:
            document.getElementById("kitchen-qual").value,

        KitchenAbvGr: Number(
            document.getElementById("kitchen-abv-gr").value
        ),


    // -----------------------------
    // Garage
    // -----------------------------

        GarageCars: Number(
            document.getElementById("garage-cars").value
        ),

        GarageQual:
            document.getElementById("garage-qual").value,

        GarageFinish:
            document.getElementById("garage-finish").value,

        GarageType:
            document.getElementById("garage-type").value,


    // -----------------------------
    // Basement
    // -----------------------------

        BsmtQual:
            document.getElementById("bsmt-qual").value,


    // -----------------------------
    // Features
    // -----------------------------

        Fireplaces: Number(
            document.getElementById("fireplaces").value
        ),

        CentralAir:
            document.getElementById("central-air").value,


    // -----------------------------
    // Property
    // -----------------------------

        LotShape:
            document.getElementById("lot-shape").value,

        MSZoning:
            document.getElementById("ms-zoning").value,

        PavedDrive:
            document.getElementById("paved-drive").value,


    // -----------------------------
    // Dates
    // -----------------------------

        YearBuilt: Number(
            document.getElementById("year-built").value
        ),

        YearRemodAdd: Number(
            document.getElementById("year-remod-add").value
        ),

        YrSold: Number(
            document.getElementById("yr-sold").value
        )

    }



// =====================================================
// SEND REQUEST TO FASTAPI
// =====================================================

    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

               body: JSON.stringify(data)
            }
        );


    // =================================================
    // HANDLE HTTP ERRORS
    // =================================================

        if (!response.ok) {

            const errorData = await response.json();

            throw new Error(
                errorData.detail ||
                "The prediction request failed."
            );
        }


    // =================================================
    // READ RESPONSE
    // =================================================

        const result = await response.json();


    // =================================================
    // DISPLAY PREDICTION
    // =================================================

        predictedPrice.textContent = formatPrice(result.predicted_sale_price);

       resultSection.hidden = false;


    } catch (error) {

        console.error("Prediction error:", error);

        errorMessage.textContent =
            error.message ||
            "Unable to connect to the prediction API.";

        errorMessage.hidden = false;


    } finally {

        // Re-enable button
        predictButton.disabled = false;
        predictButton.textContent = "Predict House Price";
    }

});

// =========================================================
// FORMAT PRICE
// =========================================================

function formatPrice(price) {

    return new Intl.NumberFormat(
        "en-US",
        {
            style: "currency",
            currency: "USD",
            maximumFractionDigits: 0
        }
    ).format(price);

}
