"""Prepare the song recommendation catalog. Requires pandas and numpy.
Run: python clean_dataset.py. Original dataset.csv is never modified.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'dataset.csv'
OUT = ROOT / 'cleaned_data'
OUT.mkdir(exist_ok=True)
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
raw = pd.read_csv(SOURCE, keep_default_na=False)
df = raw.drop(columns=['Unnamed: 0']).copy()
text_cols = ['track_id', 'track_name', 'artists', 'album_name', 'track_genre']
for c in text_cols:
    # Preserve names such as 'NA' and 'Null'; only empty/whitespace cells are missing.
    df[c] = df[c].astype('string').str.strip().replace('', pd.NA)
audio = ['danceability','energy','valence','acousticness','instrumentalness',
         'speechiness','tempo','loudness','liveness','duration_ms']
numeric = audio + ['popularity']
for c in numeric:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['explicit'] = df.explicit.astype(str).str.lower().map({'true':True,'false':False})
exact_duplicates = int(df.duplicated().sum())
df = df.drop_duplicates()
# Essential metadata and measurable audio are required for this first catalog.
flags = pd.DataFrame(index=df.index)
flags['missing_metadata'] = df[text_cols].isna().any(axis=1)
flags['invalid_track_id'] = ~df.track_id.str.fullmatch(r'[A-Za-z0-9]{22}').fillna(False)
flags['nonfinite_numeric'] = ~np.isfinite(df[numeric].to_numpy()).all(axis=1)
flags['invalid_explicit'] = df.explicit.isna()
flags['nonpositive_duration'] = df.duration_ms.le(0)
flags['unknown_or_nonpositive_tempo'] = df.tempo.le(0)
flags['out_of_range_popularity'] = ~df.popularity.between(0,100)
for c in ['danceability','energy','valence','acousticness','instrumentalness','speechiness','liveness']:
    flags['out_of_range_'+c] = ~df[c].between(0,1)
rejected = df.loc[flags.any(axis=1)].copy()
rejected['exclusion_reason'] = flags.loc[flags.any(axis=1)].apply(lambda r:'; '.join(r.index[r]),axis=1)
rejected.to_csv(OUT/'excluded_records.csv',index_label='source_row_zero_based')
valid = df.loc[~flags.any(axis=1)].copy()
stable = ['track_name','artists','album_name','explicit']+audio
conflicts = valid.groupby('track_id')[stable].nunique(dropna=False)
assert not conflicts.gt(1).any().any(), 'Unexpected metadata/audio conflicts: review before collapsing.'
groups = valid.groupby('track_id',sort=True)
clean = groups[stable].first()
# Distinct score values avoid overweighting scores repeated across genre rows.
clean['popularity'] = groups.popularity.agg(lambda x:float(np.median(x.unique())))
clean['track_genre'] = groups.track_genre.agg(lambda x:';'.join(sorted(set(x))))
clean = clean.reset_index()
clean['spotify_url'] = 'https://open.spotify.com/track/' + clean.track_id
columns = ['track_id','track_name','artists','album_name','track_genre','spotify_url',
           'popularity','explicit'] + audio
clean = clean[columns]
clean.to_csv(OUT/'songs_clean.csv',index=False,encoding='utf-8')
# Read back the actual delivered file and verify its contract.
saved = pd.read_csv(OUT/'songs_clean.csv',keep_default_na=False)
assert saved.track_id.is_unique and not saved.isna().any().any()
assert not saved[text_cols+['spotify_url']].eq('').any().any()
assert np.isfinite(saved[numeric].to_numpy()).all()
assert (saved.duration_ms>0).all() and (saved.tempo>0).all()
expected_genres = groups.track_genre.agg(lambda x:set(x)).to_dict()
assert all(set(r.track_genre.split(';'))==expected_genres[r.track_id] for r in saved.itertuples())
assert len(clean)==valid.track_id.nunique()
assert set(clean.track_id)==set(valid.track_id)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
stats = {'source_rows':len(raw),'exact_duplicates_removed':exact_duplicates,
         'excluded_rows_after_exact_dedup':len(rejected),
         'excluded_unique_track_ids':rejected.track_id.nunique(),
         'additional_rows_collapsed_by_track_id':len(valid)-len(clean),
         'clean_rows':len(clean),'clean_columns':len(clean.columns),
         'tracks_with_multiple_genres':int(clean.track_genre.str.contains(';').sum()),
         'popularity_conflicts_resolved':int(groups.popularity.nunique().gt(1).sum()),
         'source_sha256':source_hash,
         'output_sha256':hashlib.sha256((OUT/'songs_clean.csv').read_bytes()).hexdigest()}
assert len(raw)==exact_duplicates+len(rejected)+stats['additional_rows_collapsed_by_track_id']+len(clean)
(OUT/'cleaning_metrics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
report = f'''# Clean song recommendation dataset

Use `songs_clean.csv` for the next recommendation-building step.

- Original rows: {len(raw):,}
- Exact duplicate rows removed after trimming text and removing the index: {exact_duplicates:,}
- Excluded rows after exact deduplication: {len(rejected):,} ({rejected.track_id.nunique():,} distinct track IDs)
- Additional repeated-ID rows consolidated: {stats['additional_rows_collapsed_by_track_id']:,}
- Final songs: {len(clean):,}, with {len(clean.columns)} columns
- Missing/blank required values, nonfinite numeric values, and duplicate track IDs in saved output: zero

## Decisions

Removed `Unnamed: 0` (saved index) and `key`, `mode`, `time_signature` (optional musical fields omitted from the initial audio-similarity model). Retained album names for display, popularity for optional ranking, explicit for filtering, and liveness/duration for optional similarity features.

Trimmed leading/trailing whitespace while preserving spelling, accents, punctuation, and case. Literal names such as NA are preserved. Empty fields are treated as missing. Records with missing essential metadata, malformed IDs, invalid numeric values, or nonpositive duration/tempo are saved in `excluded_records.csv` with reasons. Zero tempo may mean unavailable analysis rather than a bad song: exclusion is a conservative first-version policy, and these records can be revisited. No audio values were imputed. Long durations and valid zeros in popularity/audio features were retained.

Consolidated by track ID, verifying that retained metadata and audio features agree before taking one copy. Preserved all genres in `track_genre` as sorted, semicolon-separated labels. Split this column into labels and use multilabel encoding or set overlap; do not encode each combined string as a new genre. Artists retain the source's semicolon-separated representation.

For {stats['popularity_conflicts_resolved']:,} tracks with conflicting popularity, used the median of distinct observed scores, avoiding genre-row weighting. This can produce fractional scores and is an aggregation choice, not a claim of current popularity. No timestamps were available to select the latest score.

Added `spotify_url` from track ID. Link availability has not been checked online. Different IDs sharing a title remain separate because they may represent different recordings or releases.

## Next processing step

Audio candidates: {', '.join(audio)}.

Use IDs, titles, artists, albums and URLs for lookup/display, not as continuous numeric features. Use genre/artist overlap as separate ranking signals and popularity/explicit as optional controls. Standardization and any learned transforms belong in the next model pipeline. If evaluating a held-out set, split first and fit transforms only on training data. The clean CSV deliberately retains original feature units.

## Reproducibility and verification

Run `python clean_dataset.py` with pandas and numpy. Verified the saved CSV's uniqueness, nonempty required fields, finite numbers, positive duration/tempo, preservation of each retained track's genre set, row-count reconciliation, and unchanged source SHA-256. See `cleaning_metrics.json` for hashes and counts.
'''
(OUT/'cleaning_report.md').write_text(report,encoding='utf-8')
print(json.dumps(stats,indent=2))
print('Validation passed. Saved:',OUT/'songs_clean.csv')
