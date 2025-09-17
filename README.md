# IMEREC & PNS–IMEREC (Streamlit)

A Streamlit app implementing:
- **IMEREC** (Improved MEREC) objective weighting with B/H/T normalization and stable transforms.
- **PNS–IMEREC** integration using **7 linguistic variables** mapped to Pythagorean Neutrosophic triples.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Formats

- **ObjectiveData (numeric)**: Alternatives × Criteria (columns named with hints like `(B)`, `(H)`, `(T)`).
- **LinguisticData (terms)**: Same shape as objective data, cells contain one of:
  `Very Low, Low, Medium Low, Medium, Medium High, High, Very High`.

Example files are in `data/`:
- `example_small.xlsx` → two sheets: ObjectiveData, LinguisticData.
- `objective_big.csv` → synthetic 5,000 × 8 dataset.
- `linguistic_big.csv` → terms matrix for the same 5,000 × 8 shape.

## Outputs

The app provides:
- IMEREC results (JSON).
- PNS–IMEREC results (JSON).
- Weights and rankings charts downloadable as PNG.
- Tables can be downloaded as CSV via the Streamlit download buttons.

