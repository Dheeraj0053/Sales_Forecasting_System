import pandas as pd

from src.data_loader import load_raw_excel


def test_load_raw_excel_maps_columns_and_parses_sales(tmp_path):
    sample = pd.DataFrame(
        {
            "State": ["Texas", "Texas"],
            "Date": ["2024-01-01", "2024-01-08"],
            "Total": ["1,200", "(350)"],
        }
    )
    file_path = tmp_path / "input.xlsx"
    sample.to_excel(file_path, index=False)

    loaded = load_raw_excel(str(file_path))

    assert list(loaded.columns) == ["state", "date", "sales"]
    assert loaded["state"].tolist() == ["Texas", "Texas"]
    assert loaded["sales"].tolist() == [1200.0, -350.0]
    assert str(loaded["date"].dtype).startswith("datetime64")
