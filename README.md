# Song recommendation dataset

Dataset preparation for a website that accepts a song and recommends songs with similar audio characteristics, genres, and artists.

## Current state

- EDA of 114,000 source rows.
- Clean catalog of 89,583 unique tracks with 18 columns.
- Spotify track links and merged genre labels.
- No recommendation model or website implemented yet.

## Run locally

Requires Python 3.10 or later.

```sh
python -m pip install -r requirements.txt
python eda.py
python clean_dataset.py
```

Open `eda_output/eda_report.html` for the EDA and `cleaned_data/cleaning_report.md` for cleaning decisions. The prepared catalog is `cleaned_data/songs_clean.csv`. Both scripts preserve `dataset.csv`.

## Data preparation

The cleaner removes the saved index and optional key/mode/time-signature fields, removes exact duplicates, consolidates tracks by ID, and preserves all genre labels separated by semicolons. Conflicting popularity scores are aggregated using the median of distinct values. Records with missing essential metadata or nonpositive duration/tempo are saved separately for review. No audio values are imputed.

Fit feature scaling and encoding in the recommendation pipeline. Use track IDs to avoid overlap when evaluating train/test sets. Metadata and URLs support lookup/display; audio features support similarity; genres and artists provide additional ranking signals.

## Data provenance

The source CSV was supplied locally. Its original source, collection date, and license have not been documented. No license is asserted for the dataset. Generated Spotify URLs have not been checked for availability.

## Saving future changes

After reviewing changes, stage the intended files, commit with a descriptive message, and push to the configured GitHub remote. Commits are created explicitly; no automatic commit schedule is configured.
