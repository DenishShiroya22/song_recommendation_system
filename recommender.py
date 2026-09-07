"""Reusable cosine k-NN song engine and command-line search."""
from pathlib import Path
import argparse
import json
import unicodedata
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models/song_knn.joblib'

def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())

class SongRecommender:
    def __init__(self, catalog, matrix):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.matrix = sparse.csr_matrix(matrix, dtype=np.float32)
        required = {'track_id','track_name','artists','track_genre','spotify_url','explicit'}
        if not required.issubset(self.catalog.columns):
            raise ValueError('Catalog is missing required metadata.')
        if len(self.catalog) != self.matrix.shape[0] or len(self.catalog) < 2:
            raise ValueError('Catalog and feature rows must align and include at least two songs.')
        if not self.catalog.track_id.is_unique:
            raise ValueError('Track IDs must be unique.')
        if not np.isfinite(self.matrix.data).all() or np.any(self.matrix.multiply(self.matrix).sum(axis=1).A1 == 0):
            raise ValueError('Features must be finite and have nonzero row norms.')
        self.explicit = self.catalog.explicit.astype(str).str.lower().map({'true':True,'false':False})
        if self.explicit.isna().any():
            raise ValueError('Explicit values must be true or false.')
        self.index = {track:i for i,track in enumerate(self.catalog.track_id)}
        self.titles = self.catalog.track_name.map(normalize)
        self.artists = self.catalog.artists.map(normalize)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1).fit(self.matrix)

    def search(self, query, limit=10):
        """Return matches for user selection; never silently pick an ambiguous title."""
        query = normalize(query)
        if not query:
            raise ValueError('Enter a song title or artist.')
        if not isinstance(limit,int) or isinstance(limit,bool) or limit < 1:
            raise ValueError('limit must be a positive integer.')
        text = self.titles + ' ' + self.artists
        mask = pd.Series(True,index=self.catalog.index)
        for token in query.split():
            mask &= text.str.contains(token,regex=False)
        matches = self.catalog.loc[mask].copy()
        matches['_exact'] = self.titles.loc[mask].eq(query)
        matches = matches.sort_values(['_exact','track_name','artists','track_id'],ascending=[False,True,True,True])
        return matches.drop(columns='_exact').head(limit).to_dict(orient='records')

    def recommend(self, track_id, k=20, exclude_explicit=False):
        """Exact nearest neighbors in combined audio+genre space; score = 1-distance."""
        if track_id not in self.index:
            raise ValueError('Song is not in this catalog. Search and select a listed track_id.')
        if not isinstance(k,int) or isinstance(k,bool) or k < 1:
            raise ValueError('k must be a positive integer.')
        row = self.index[track_id]
        count = min(len(self.catalog),k+1)
        # Expand only when filtering removes neighbors; no full pairwise matrix is built.
        while True:
            distances, indices = self.model.kneighbors(self.matrix[row],n_neighbors=count)
            candidates = [(float(d),int(i)) for d,i in zip(distances[0],indices[0])
                          if i != row and (not exclude_explicit or not self.explicit.iloc[i])]
            if len(candidates) >= k or count == len(self.catalog):
                break
            count = min(len(self.catalog),count*2)
        candidates.sort(key=lambda item:(item[0],self.catalog.iloc[item[1]].track_id))
        seed_genres = set(self.catalog.iloc[row].track_genre.split(';'))
        seed_artists = set(self.artists.iloc[row].split(';'))
        results = []
        for distance,i in candidates[:k]:
            record = self.catalog.iloc[i].to_dict()
            record['cosine_distance'] = float(np.clip(distance,0,2))
            record['similarity'] = 1-record['cosine_distance']
            record['shared_genres'] = sorted(seed_genres & set(record['track_genre'].split(';')))
            record['same_artist'] = bool(seed_artists & set(self.artists.iloc[i].split(';')))
            results.append(record)
        return results

    def save(self,path):
        # Store plain data and sklearn objects; avoid serializing a __main__ class.
        joblib.dump({'catalog':self.catalog,'matrix':self.matrix,'model':self.model},path,compress=3)

    @classmethod
    def load(cls,path=DEFAULT_MODEL):
        # Only load trusted local joblib files.
        artifact = joblib.load(path)
        instance = cls(artifact['catalog'],artifact['matrix'])
        if artifact['model'].metric != 'cosine' or artifact['model'].n_samples_fit_ != len(instance.catalog):
            raise ValueError('Model artifact does not match catalog.')
        instance.model = artifact['model']
        return instance

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--search',help='Search title and/or artist; returns candidate IDs.')
    actions.add_argument('--track-id',help='Selected song ID for recommendations.')
    parser.add_argument('--k',type=int,default=20)
    parser.add_argument('--exclude-explicit',action='store_true')
    args = parser.parse_args()
    engine = SongRecommender.load(args.model)
    try:
        results = engine.search(args.search,args.k) if args.search is not None else engine.recommend(args.track_id,args.k,args.exclude_explicit)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()

