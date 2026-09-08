# Song recommendation engine

Update: the website now uses separately normalized sound/genre scores with a user-controlled weight (default 70/30), and search supports misspellings. The combined-cosine model described below remains available as the legacy baseline when audio_weight is omitted. See ../INTEGRATIONS.md for the new formula, feedback evaluation, and Spotify export setup.

The saved engine uses sklearn.neighbors.NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1). Fitting stores the catalog feature vectors. Each request calculates the nearest neighbors; no regression/classification target or static predictive formula is learned.

## Run

From the project root with the project environment:

~~~powershell
./.venv/Scripts/python.exe train_recommender.py
./.venv/Scripts/python.exe recommender.py --search "Comedy Gen Hoshino" --k 5
./.venv/Scripts/python.exe recommender.py --track-id 5SuOikwiRyPMVoIQDJUgSV --k 20
./.venv/Scripts/python.exe recommender.py --track-id 5SuOikwiRyPMVoIQDJUgSV --k 20 --exclude-explicit
./.venv/Scripts/python.exe -m unittest test_recommender.py test_feature_engineering.py -v
~~~

Search returns matching titles/artists and IDs. Have the website user select the intended recording before requesting recommendations. Search is case-insensitive Unicode-normalized token matching, not typo-tolerant fuzzy search. It searches the local catalog only.

## Website integration

~~~python
from recommender import SongRecommender
engine = SongRecommender.load()  # once when the server starts
matches = engine.search("Comedy Gen Hoshino", limit=5)
playlist = engine.recommend(matches[0]["track_id"], k=20)
~~~

In the actual UI, use the user's selected ID instead of automatically taking the first match. Returned dictionaries contain song metadata, Spotify URL, cosine distance, similarity, shared genres, and a same-artist flag. No website/API server is included yet.

The input song is excluded by ID. Results have unique IDs, but alternate releases with different IDs can still appear. Explicit filtering is optional and may return fewer than k results if insufficient eligible songs exist. No artist-diversity cap or artist boost is applied.

## Feature and score meaning

The index uses the existing 122-column matrix: 8 standardized audio columns and 114 multi-hot genre columns. No PCA or additional weights are applied. Popularity, explicit, duration and artist identity do not enter cosine similarity. Explicit is a filter; shared artists are explanatory metadata.

Similarity = 1 - cosine distance, theoretically in [-1, 1]. It is not a probability or an accuracy percentage. Standardized audio can be negative. Different genre counts and audio-vector norms affect the combined score; separate audio/genre weighting is a later tuning experiment.

## Files and verification

- song_knn.joblib: catalog, sparse features and fitted sklearn neighbor index; load only trusted joblib artifacts.
- training_report.json: dataset hashes, dimensions, fitting and query timings, validation scope.
- example_recommendations.json: actual output for one selected song.

The build verifies the source hash and joins catalog metadata to the feature-row mapping by ID. It checks 10 seeded queries against independently calculated cosine similarities, self-exclusion, unique IDs, score ordering and save/load agreement. Unit tests cover ambiguity, invalid inputs, filtering, identical vectors, constant features and preprocessing behavior.

No N-by-N similarity matrix is stored. Brute-force retrieval scans the catalog per query. Reported latency is a small local measurement, not a production benchmark. These checks establish numerical correctness, not listener satisfaction. Relevance needs listening feedback or labeled evaluation. Rebuild features and the index together when updating the catalog.

