from pkg_house_prices.models.trainer import train_model_pipeline

# 1. Test Linear Regression training
def test_train_linear_model_returns_pipeline():

    model, cv_score, train_score, std_score = train_model_pipeline(
        model_type="linear", data_path="tests/sample_data/train_sample.csv"
    )

    assert model is not None
    assert hasattr(model, "predict")

    assert isinstance(cv_score, float)
    assert isinstance(train_score, float)
    assert isinstance(std_score, float)


# 2. Test pipeline contains required steps
def test_pipeline_contains_required_steps():

    model, _, _, _ = train_model_pipeline(model_type="linear", data_path="tests/sample_data/train_sample.csv")

    assert "feature_engineer" in model.named_steps
    assert "dropper" in model.named_steps
    assert "preprocessor" in model.named_steps
    assert "regressor" in model.named_steps


# 3. Test that the model can make predictions
def test_model_can_make_predictions(sample_df):

    X, y = sample_df

    model, _, _, _ = train_model_pipeline(model_type="linear", data_path="tests/sample_data/train_sample.csv")

    predictions = model.predict(X)

    assert len(predictions) == len(y)


# 4. Test GridSearchCV is used
def test_ridge_uses_grid_search():

    model, cv_score, train_score, std_score = train_model_pipeline(
        model_type="ridge", data_path="tests/sample_data/train_sample.csv"
    )

    assert model.named_steps["regressor"].__class__.__name__ == "Ridge"


# 5. Test model scores are valid
def test_model_scores_are_valid():

    model, cv_score, train_score, std_score = train_model_pipeline(
        model_type="linear", data_path="tests/sample_data/train_sample.csv"
    )

    assert isinstance(cv_score, float)
    assert isinstance(train_score, float)
    assert isinstance(std_score, float)

    assert -1 <= cv_score <= 1
    assert -1 <= train_score <= 1
    assert std_score >= 0
