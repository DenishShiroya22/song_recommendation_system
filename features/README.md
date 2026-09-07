# Engineered song features

The catalog has 89,583 rows. Eight audio features are standardized with StandardScaler, and each individual genre has a binary column. This is multi-hot encoding because a song can belong to multiple genres.

Audio: danceability, energy, valence, acousticness, instrumentalness, speechiness, tempo, loudness.

Popularity, explicit, duration_ms, liveness, and metadata are excluded from the initial similarity matrix. They remain in the clean dataset. Artist overlap belongs in the next ranking stage.

## Files

- song_features.npz: combined float32 CSR matrix; load with scipy.sparse.load_npz.
- audio_scaled.npy: standardized audio; load with numpy.load.
- genre_encoded.npz: binary genre features, unscaled.
- track_index.csv: zero-based feature_row and track_id mapping; join metadata by ID.
- feature_names.json: exact combined column order.
- preprocessor.joblib: fitted scaler and genre encoder. Load only trusted artifacts.
- scaler_parameters.csv: fitted mean, population variance, and standard deviation.
- manifest.json: dimensions, source hash, fit scope, and validation.

## Reuse

Run python feature_engineering.py from the project environment. Load preprocessor.joblib with joblib.load, then call transform_features(frame, state) from feature_engineering.py. It returns audio, genre, and combined matrices.

Do not refit on a user's selected song. Look up its existing matrix row by track_id. Unknown genres raise an explicit error rather than silently disappearing.

StandardScaler applies (value - fitted mean) / fitted standard deviation with ddof=0. Values are not restricted to [0,1]. Constant columns become zero.

## Evaluation and ranking

Default fitting uses the whole available catalog for unsupervised retrieval; this is not a held-out evaluation. Split first for evaluation and pass --fit-track-ids training_ids.csv, containing a track_id column. The pipeline fits only these rows and transforms the entire input. Any unseen validation genre needs an explicit policy.

The combined matrix has no custom weights. Scaling audio does not establish the desired balance between genre and audio similarity. Use the separate matrices to tune that balance in the recommender. Previously discussed 70/20/10 weights are not applied or validated here. No recommendation model has been trained.

