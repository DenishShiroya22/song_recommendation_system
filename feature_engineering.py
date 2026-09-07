"""Song features: python feature_engineering.py [--fit-track-ids training_ids.csv]."""
from pathlib import Path
import argparse, hashlib, json
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

ROOT = Path(__file__).resolve().parent
AUDIO = ['danceability','energy','valence','acousticness','instrumentalness','speechiness','tempo','loudness']

def labels(frame):
    result = [sorted(set(p.strip() for p in str(v).split(';') if p.strip())) for v in frame.track_genre]
    if any(not row for row in result):
        raise ValueError('Every song needs a genre.')
    return result

def validate(frame):
    required = ['track_id','track_genre'] + AUDIO
    if not set(required).issubset(frame.columns):
        raise ValueError('Required columns are missing.')
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Empty input or missing required values.')
    if not frame.track_id.is_unique or frame.track_id.astype(str).str.strip().eq('').any():
        raise ValueError('Track IDs must be nonempty and unique.')
    if not np.isfinite(frame[AUDIO].to_numpy(dtype=float)).all():
        raise ValueError('Audio must be finite numeric values.')

def fit_preprocessor(frame):
    validate(frame)
    return {'audio_columns': AUDIO.copy(),
            'scaler': StandardScaler().fit(frame[AUDIO]),
            'genres': MultiLabelBinarizer(sparse_output=True).fit(labels(frame))}

def transform_features(frame, state):
    validate(frame)
    rows = labels(frame)
    unknown = set(g for row in rows for g in row) - set(state['genres'].classes_)
    if unknown:
        raise ValueError(f'Unknown genres: {sorted(unknown)}. Refit deliberately or define an unknown-genre policy.')
    audio = state['scaler'].transform(frame[state['audio_columns']]).astype(np.float32)
    genre = state['genres'].transform(rows).astype(np.float32).tocsr()
    combined = sparse.hstack([sparse.csr_matrix(audio),genre],format='csr',dtype=np.float32)
    return audio,genre,combined

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'cleaned_data/songs_clean.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'features')
    parser.add_argument('--fit-track-ids',type=Path,help='CSV with training track_id values; transforms all input rows.')
    args = parser.parse_args()
    frame = pd.read_csv(args.input,keep_default_na=False)
    validate(frame)
    fit_frame = frame
    if args.fit_track_ids:
        ids = pd.read_csv(args.fit_track_ids,keep_default_na=False).track_id
        if ids.empty or not ids.is_unique or not set(ids).issubset(set(frame.track_id)):
            raise ValueError('Fit IDs must be unique, nonempty and present in input.')
        fit_frame = frame.loc[frame.track_id.isin(ids)]
    state = fit_preprocessor(fit_frame)
    audio,genre,combined = transform_features(frame,state)
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    names = AUDIO + ['genre__'+g for g in state['genres'].classes_]
    joblib.dump(state,out/'preprocessor.joblib')
    np.save(out/'audio_scaled.npy',audio,allow_pickle=False)
    sparse.save_npz(out/'genre_encoded.npz',genre)
    sparse.save_npz(out/'song_features.npz',combined)
    frame[['track_id']].reset_index(names='feature_row').to_csv(out/'track_index.csv',index=False)
    (out/'feature_names.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
    scaler = state['scaler']
    pd.DataFrame({'feature':AUDIO,'mean':scaler.mean_,'variance':scaler.var_,'scale':scaler.scale_}).to_csv(out/'scaler_parameters.csv',index=False)
    reloaded = joblib.load(out/'preprocessor.joblib')
    a,g,x = transform_features(frame,reloaded)
    assert (sparse.load_npz(out/'song_features.npz') != x).nnz == 0
    assert (sparse.load_npz(out/'genre_encoded.npz') != g).nnz == 0
    assert np.array_equal(np.load(out/'audio_scaled.npy'),a)
    assert np.isfinite(x.data).all()
    assert x.shape == (len(frame),len(names))
    assert all(set(a)==set(b) for a,b in zip(state['genres'].inverse_transform(genre),labels(frame)))
    assert pd.read_csv(out/'track_index.csv').track_id.tolist()==frame.track_id.tolist()
    fitted = scaler.transform(fit_frame[AUDIO])
    np.testing.assert_allclose(fitted.mean(axis=0),0,atol=1e-10)
    np.testing.assert_allclose(fitted.std(axis=0)[scaler.var_>0],1,atol=1e-10)
    manifest = {'rows':len(frame),'fit_rows':len(fit_frame),'audio_features':len(AUDIO),
                'genre_features':genre.shape[1],'combined_features':combined.shape[1],
                'dtype':'float32','format':'CSR sparse matrix',
                'fit_scope':'training IDs' if args.fit_track_ids else 'full catalog for content-based retrieval',
                'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'sklearn_version':sklearn.__version__,
                'validation':'Saved matrices, all genre memberships, scaler statistics, and row mapping verified.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()

