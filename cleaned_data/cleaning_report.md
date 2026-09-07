# Clean song recommendation dataset

Use `songs_clean.csv` for the next recommendation-building step.

- Original rows: 114,000
- Exact duplicate rows removed after trimming text and removing the index: 450
- Excluded rows after exact deduplication: 158 (158 distinct track IDs)
- Additional repeated-ID rows consolidated: 23,809
- Final songs: 89,583, with 18 columns
- Missing/blank required values, nonfinite numeric values, and duplicate track IDs in saved output: zero

## Decisions

Removed `Unnamed: 0` (saved index) and `key`, `mode`, `time_signature` (optional musical fields omitted from the initial audio-similarity model). Retained album names for display, popularity for optional ranking, explicit for filtering, and liveness/duration for optional similarity features.

Trimmed leading/trailing whitespace while preserving spelling, accents, punctuation, and case. Literal names such as NA are preserved. Empty fields are treated as missing. Records with missing essential metadata, malformed IDs, invalid numeric values, or nonpositive duration/tempo are saved in `excluded_records.csv` with reasons. Zero tempo may mean unavailable analysis rather than a bad song: exclusion is a conservative first-version policy, and these records can be revisited. No audio values were imputed. Long durations and valid zeros in popularity/audio features were retained.

Consolidated by track ID, verifying that retained metadata and audio features agree before taking one copy. Preserved all genres in `track_genre` as sorted, semicolon-separated labels. Split this column into labels and use multilabel encoding or set overlap; do not encode each combined string as a new genre. Artists retain the source's semicolon-separated representation.

For 720 tracks with conflicting popularity, used the median of distinct observed scores, avoiding genre-row weighting. This can produce fractional scores and is an aggregation choice, not a claim of current popularity. No timestamps were available to select the latest score.

Added `spotify_url` from track ID. Link availability has not been checked online. Different IDs sharing a title remain separate because they may represent different recordings or releases.

## Next processing step

Audio candidates: danceability, energy, valence, acousticness, instrumentalness, speechiness, tempo, loudness, liveness, duration_ms.

Use IDs, titles, artists, albums and URLs for lookup/display, not as continuous numeric features. Use genre/artist overlap as separate ranking signals and popularity/explicit as optional controls. Standardization and any learned transforms belong in the next model pipeline. If evaluating a held-out set, split first and fit transforms only on training data. The clean CSV deliberately retains original feature units.

## Reproducibility and verification

Run `python clean_dataset.py` with pandas and numpy. Verified the saved CSV's uniqueness, nonempty required fields, finite numbers, positive duration/tempo, preservation of each retained track's genre set, row-count reconciliation, and unchanged source SHA-256. See `cleaning_metrics.json` for hashes and counts.
