import unittest
import numpy as np
import pandas as pd
from feature_engineering import AUDIO, fit_preprocessor, transform_features

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({c:[0.2,0.8] for c in AUDIO})
        self.frame['track_id'] = ['a','b']
        self.frame['track_genre'] = ['rock;pop;rock','pop']
    def test_multigenre_and_scaling(self):
        state = fit_preprocessor(self.frame)
        a,g,x = transform_features(self.frame,state)
        np.testing.assert_array_equal(g.toarray(),[[1,1],[1,0]])
        np.testing.assert_allclose(a,[[-1]*8,[1]*8])
        self.assertEqual(x.shape,(2,10))
    def test_reuses_training_statistics(self):
        state = fit_preprocessor(self.frame)
        new = self.frame.iloc[:1].copy()
        new[AUDIO] = 1.4
        a,_,_ = transform_features(new,state)
        np.testing.assert_allclose(a,[[3]*8],atol=1e-6)
    def test_unknown_genre(self):
        state = fit_preprocessor(self.frame)
        self.frame.loc[0,'track_genre'] = 'unknown'
        with self.assertRaisesRegex(ValueError,'Unknown'):
            transform_features(self.frame,state)
    def test_duplicate_ids(self):
        self.frame['track_id'] = 'same'
        with self.assertRaisesRegex(ValueError,'unique'):
            fit_preprocessor(self.frame)
    def test_constant_feature(self):
        self.frame['tempo'] = 120
        state = fit_preprocessor(self.frame)
        a,_,_ = transform_features(self.frame,state)
        np.testing.assert_array_equal(a[:,AUDIO.index('tempo')],[0,0])

if __name__ == '__main__':
    unittest.main()

