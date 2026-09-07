"""Build the full-catalog nearest-neighbor index and run retrieval checks."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
from recommender import SongRecommender

ROOT = Path(__file__).resolve().parent

def main():
    catalog_path = ROOT/'cleaned_data/songs_clean.csv'
    features = ROOT/'features'
    manifest = json.loads((features/'manifest.json').read_text(encoding='utf-8-sig'))
    if hashlib.sha256(catalog_path.read_bytes()).hexdigest() != manifest['input_sha256']:
        raise ValueError('Catalog changed since feature generation. Regenerate features first.')
    catalog = pd.read_csv(catalog_path,keep_default_na=False)
    mapping = pd.read_csv(features/'track_index.csv',keep_default_na=False)
    matrix = sparse.load_npz(features/'song_features.npz')
    if not mapping.track_id.is_unique or set(mapping.track_id) != set(catalog.track_id):
        raise ValueError('Feature track IDs do not match catalog.')
    if mapping.feature_row.tolist() != list(range(len(mapping))):
        raise ValueError('Feature rows are not in contiguous matrix order.')
    catalog = catalog.set_index('track_id').loc[mapping.track_id].reset_index()
    names = json.loads((features/'feature_names.json').read_text(encoding='utf-8-sig'))
    if matrix.shape != (len(catalog),len(names)):
        raise ValueError('Feature dimensions do not match metadata.')
    start = time.perf_counter()
    engine = SongRecommender(catalog,matrix)
    fit_seconds = time.perf_counter()-start
    out = ROOT/'models'
    out.mkdir(exist_ok=True)
    engine.save(out/'song_knn.joblib')
    loaded = SongRecommender.load(out/'song_knn.joblib')
    timings = []
    for row in np.random.default_rng(42).choice(len(catalog),size=10,replace=False):
        track_id = catalog.iloc[row].track_id
        start = time.perf_counter()
        results = loaded.recommend(track_id,20)
        timings.append(time.perf_counter()-start)
        ids = [r['track_id'] for r in results]
        assert len(ids)==20 and len(set(ids))==20 and track_id not in ids
        scores = [r['similarity'] for r in results]
        assert all(a>=b-1e-6 for a,b in zip(scores,scores[1:]))
        # Independent direct similarity for one query at a time; never N by N.
        direct = cosine_similarity(matrix[row],matrix).ravel()
        direct[row] = -np.inf
        expected = np.sort(direct)[-20:][::-1]
        np.testing.assert_allclose(scores,expected,atol=2e-6)
        old = engine.recommend(track_id,20)
        assert [r['track_id'] for r in old]==ids
    sample_id = engine.search('Comedy Gen Hoshino',1)[0]['track_id']
    sample = {'input':catalog.loc[catalog.track_id.eq(sample_id)].iloc[0].to_dict(),
              'recommendations':loaded.recommend(sample_id,10)}
    (out/'example_recommendations.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2),encoding='utf-8')
    report = {'songs':len(catalog),'features':matrix.shape[1],'algorithm':'brute-force k-nearest neighbors',
              'metric':'cosine','sklearn_version':sklearn.__version__,'fit_seconds':fit_seconds,
              'median_query_seconds_10_samples':float(np.median(timings)),
              'validation':'10 seeded queries: exact cosine agreement, self-exclusion, unique results, ordering, save/load agreement.',
              'scope':'Full-catalog instance-based retrieval; no supervised target or held-out relevance labels.',
              'catalog_sha256':manifest['input_sha256'],
              'feature_sha256':hashlib.sha256((features/'song_features.npz').read_bytes()).hexdigest()}
    (out/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()

