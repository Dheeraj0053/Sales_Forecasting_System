import pandas as pd

from src.preprocessing import preprocess_sales_data


def test_preprocess_resamples_weekly_and_fills_missing():
    raw = pd.DataFrame(
        {
            "state": ["CA", "CA", "CA"],
            "date": pd.to_datetime(["2024-01-07", "2024-01-21", "2024-01-28"]),
            "sales": [100.0, None, 200.0],
        }
    )

    processed = preprocess_sales_data(raw)
    ca = processed[processed["state"] == "CA"].sort_values("date").reset_index(drop=True)

    assert len(ca) == 4
    assert ca["sales"].isna().sum() == 0
    assert ca.iloc[1]["sales"] >= 0
