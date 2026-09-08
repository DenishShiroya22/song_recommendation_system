import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from recommender import SongRecommender

class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame({'track_id':['a','b','c','d'],'track_name':['Song','Song','Other','Last'],
            'artists':['Artist','Another','Artist','Other'],'track_genre':['pop','pop;rock','rock','jazz'],
            'spotify_url':['https://open.spotify.com/track/'+i for i in 'abcd'],
            'explicit':[False,True,False,False]})
        self.engine = SongRecommender(self.catalog,sparse.csr_matrix([[1,0],[1,0],[0.8,0.6],[-1,0]],dtype=np.float32))
    def test_exact_cosine_and_self_exclusion(self):
        rows = self.engine.recommend('a',3)
        self.assertEqual([r['track_id'] for r in rows],['b','c','d'])
        np.testing.assert_allclose([r['similarity'] for r in rows],[1,.8,-1],atol=1e-6)
    def test_filter_and_large_k(self):
        rows = self.engine.recommend('a',10,exclude_explicit=True)
        self.assertEqual([r['track_id'] for r in rows],['c','d'])
    def test_search_returns_ambiguous_matches(self):
        self.assertEqual(len(self.engine.search(' SONG ')),2)
        self.assertEqual(self.engine.search('song another')[0]['track_id'],'b')
        self.assertEqual(self.engine.search('nonexistent'),[])
    def test_bad_queries(self):
        for track,k in [('unknown',1),('a',0)]:
            with self.assertRaises(ValueError):
                self.engine.recommend(track,k)
        with self.assertRaises(ValueError):
            self.engine.search(' ')
    def test_search_popularity_precedes_alphabetical_order_and_limit(self):
        catalog = self.catalog.copy()
        catalog['track_name'] = ['Song', 'Song', 'Song Remix', 'Song Acoustic']
        catalog['popularity'] = [86, 24, 99, None]
        engine = SongRecommender(catalog, self.engine.matrix)
        self.assertEqual([r['track_id'] for r in engine.search('song', 4)], ['a','b','c','d'])
        self.assertEqual(engine.search('song', 1)[0]['track_id'], 'a')
        self.assertEqual([r['track_id'] for r in engine.search('artist', 2)], ['c','a'])

    def test_fuzzy_popularity_ties_consider_all_candidates(self):
        catalog = pd.concat([self.catalog.iloc[[0]]] * 12, ignore_index=True)
        catalog['track_id'] = [f't{i:02}' for i in range(12)]
        catalog['track_name'] = 'Shape of You'
        catalog['popularity'] = list(range(12))
        engine = SongRecommender(catalog, sparse.csr_matrix(np.ones((12, 2))))
        self.assertEqual(engine.search('shpe of you', 1)[0]['track_id'], 't11')
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.joblib'
            self.engine.save(path)
            self.assertEqual(SongRecommender.load(path).recommend('a',3),self.engine.recommend('a',3))
    def test_zero_vectors_rejected(self):
        with self.assertRaises(ValueError):
            SongRecommender(self.catalog,sparse.csr_matrix((4,2)))

if __name__ == '__main__':
    unittest.main()

