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

