console.log("My script.js loaded");

const form = document.getElementById("prediction-form");
const resultSection = document.getElementById("prediction-result");
const predictedPrice = document.getElementById("predicted-price");
const errorMessage = document.getElementById("error-message");
const predictButton = document.getElementById("predict-button");

console.log("form =", form);
console.log("button =", predictButton);

// =========================================================
// CHECK REQUIRED ELEMENTS
// =========================================================

if (!form) {
    console.error("ERROR: prediction-form was not found.");
}

if (!predictButton) {
    console.error("ERROR: predict-button was not found.");
}


// =========================================================
// FORM SUBMISSION
// =========================================================

console.log("ABOUT TO REGISTER SUBMIT HANDLER");

form.addEventListener("submit", async function (event) {

    // Prevent the browser from reloading the page
    event.preventDefault();

    console.log("SUBMIT EVENT FIRED");

    // =====================================================
    // CLEAR PREVIOUS MESSAGES
    // =====================================================

    resultSection.hidden = true;
    errorMessage.hidden = true;

    // Clear previous error text
    errorMessage.textContent = "";

    // =====================================================
    // DISABLE BUTTON WHILE PREDICTION IS RUNNING
    // =====================================================

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
    };


    // =====================================================
    // LOG DATA BEING SENT TO API
    // =====================================================

    console.log("Data being sent to /predict:", data);


    // =====================================================
    // CHECK FOR INVALID NUMERIC VALUES
    // =====================================================

    const numericFields = [
        "OverallQual",
        "OverallCond",
        "TotalBsmtSF",
        "1stFlrSF",
        "2ndFlrSF",
        "GrLivArea",
        "FullBath",
        "HalfBath",
        "BsmtFullBath",
        "BsmtHalfBath",
        "KitchenAbvGr",
        "GarageCars",
        "Fireplaces",
        "YearBuilt",
        "YearRemodAdd",
        "YrSold"
    ];

    const invalidFields = numericFields.filter(
        field => Number.isNaN(data[field])
    );

    if (invalidFields.length > 0) {

        console.error(
            "Invalid numeric fields:",
            invalidFields
        );

        errorMessage.textContent =
            "Please enter valid values for: " +
            invalidFields.join(", ");

        errorMessage.hidden = false;

        predictButton.disabled = false;
        predictButton.textContent = "Predict House Price";

        return;
    }


    // =====================================================
    // SEND REQUEST TO FASTAPI
    // =====================================================

    try {

        console.log("Sending POST request to /predict...");

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        // =================================================
        // LOG RESPONSE STATUS
        // =================================================

        console.log(
            "API response status:",
            response.status,
            response.statusText
        );


        // =================================================
        // HANDLE HTTP ERRORS
        // =================================================

        if (!response.ok) {

            const contentType =
                response.headers.get("content-type") || "";

            let errorMessageText =
                "The prediction request failed.";

            // FastAPI normally returns JSON for validation errors
            if (contentType.includes("application/json")) {

                const errorData = await response.json();

                console.error(
                    "API JSON error:",
                    errorData
                );

                // FastAPI validation errors
                if (Array.isArray(errorData.detail)) {

                    errorMessageText =
                        errorData.detail
                            .map(error => {

                                const location =
                                    error.loc
                                        ? error.loc.join(".")
                                        : "field";

                                return `${location}: ${error.msg}`;
                            })
                            .join(" | ");
                }

                // Normal FastAPI error
                else if (errorData.detail) {

                    errorMessageText =
                        errorData.detail;
                }
            }

            // Render may return plain text or HTML
            else {

                const errorText =
                    await response.text();

                console.error(
                    "API non-JSON error:",
                    errorText
                );

                if (errorText.trim()) {

                    errorMessageText =
                        errorText;
                }
            }

            throw new Error(
                `API error ${response.status}: ${errorMessageText}`
            );
        }


        // =================================================
        // READ SUCCESS RESPONSE
        // =================================================

        const contentType =
            response.headers.get("content-type") || "";

        if (!contentType.includes("application/json")) {

            const responseText =
                await response.text();

            console.error(
                "Unexpected non-JSON success response:",
                responseText
            );

            throw new Error(
                "The API returned an unexpected response."
            );
        }

        const result =
            await response.json();

        console.log(
            "Prediction response:",
            result
        );


        // =================================================
        // CHECK PREDICTION VALUE
        // =================================================

        if (
            result.predicted_sale_price === undefined ||
            result.predicted_sale_price === null
        ) {

            console.error(
                "Prediction field missing:",
                result
            );

            throw new Error(
                "The API response does not contain a predicted sale price."
            );
        }


        // =================================================
        // DISPLAY PREDICTION
        // =================================================

        predictedPrice.textContent =
            formatPrice(
                result.predicted_sale_price
            );

        resultSection.hidden = false;

        console.log(
            "Prediction displayed successfully."
        );
    }


    // =====================================================
    // HANDLE ERRORS
    // =====================================================

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        errorMessage.textContent =
            error.message ||
            "Unable to connect to the prediction API.";

        errorMessage.hidden = false;
    }


    // =====================================================
    // RE-ENABLE BUTTON
    // =====================================================

    finally {

        predictButton.disabled = false;
        predictButton.textContent =
            "Predict House Price";
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
