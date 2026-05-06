from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(seed=42)
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    states = ["California", "Texas", "Florida"]
    dates = pd.date_range("2021-01-03", periods=110, freq="W")
    rows = []
    for state_idx, state in enumerate(states):
        baseline = 1000 + state_idx * 250
        trend = np.linspace(0, 180, len(dates))
        seasonal = 60 * np.sin(np.arange(len(dates)) * 2 * np.pi / 52)
        noise = rng.normal(0, 25, len(dates))
        sales = baseline + trend + seasonal + noise
        for d, y in zip(dates, sales):
            rows.append({"State": state, "Date": d, "Total": round(float(y), 2)})

    df = pd.DataFrame(rows)
    df.to_excel(data_dir / "raw.xlsx", index=False)
    print("Sample data created at data/raw.xlsx")


if __name__ == "__main__":
    main()
