import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS, add_time_features


def test_feature_engineering_adds_expected_columns_without_leakage():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "sales": list(range(1, 41)),
        }
    )

    features = add_time_features(df)
    for column in FEATURE_COLUMNS:
        assert column in features.columns

    # lag_1 at row 10 should equal sales from row 9
    assert features.loc[10, "lag_1"] == features.loc[9, "sales"]
    # rolling features are based on shifted history, not current value
    assert features.loc[20, "roll_mean_7"] != features.loc[20, "sales"]
