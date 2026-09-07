# Song recommendation dataset

Dataset preparation for a website that accepts a song and recommends songs with similar audio characteristics, genres, and artists.

## Current state

- EDA of 114,000 source rows.
- Clean catalog of 89,583 unique tracks with 18 columns.
- Spotify track links and merged genre labels.
- Cosine k-nearest-neighbor engine with a Streamlit song-search and playlist interface.

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

## Feature engineering

Run `python feature_engineering.py` to generate standardized audio and multi-hot genre features. Run `python -m unittest test_feature_engineering.py` to check preprocessing behavior. See `features/README.md` for the saved matrices, row mapping, reuse instructions and training-only fit option. The default fits the entire catalog for content-based retrieval.


## Recommendation engine

Run `python train_recommender.py` to build the cosine k-NN index, then `python recommender.py --search "Comedy Gen Hoshino" --k 5` to find a song. Pass the selected ID using `--track-id ID --k 20` to generate recommendations. See `models/README.md` for commands, Python integration, score interpretation, and validation.


## Streamlit website

Start with `python -m streamlit run app.py` (on this Windows setup: `./.venv/Scripts/python.exe -m streamlit run app.py`). Open http://localhost:8501. Search by song/artist, select a recording, choose 5–50 recommendations, optionally exclude explicit tracks, and generate the playlist. Each recommendation opens in Spotify; playlist links can be downloaded as text. Spotify-account playlist saving is not connected.

The app caches the model between requests and rebuilds the cache when the saved model changes. Run `python -m unittest test_app.py -v` to verify the website flow.
