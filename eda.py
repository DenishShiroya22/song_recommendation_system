"""Reproducible, read-only Spotify EDA. Run: python eda.py [dataset.csv].
Dependencies: pandas, numpy. Outputs: eda_output/ (HTML report and audit tables).
"""
from pathlib import Path
import sys, json, hashlib, html
import numpy as np
import pandas as pd

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'dataset.csv'
OUT = SOURCE.parent / 'eda_output'
OUT.mkdir(exist_ok=True)
df = pd.read_csv(SOURCE)
data = df.drop(columns=['Unnamed: 0'], errors='ignore')
features = ['popularity','duration_ms','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
numeric = data.select_dtypes(include='number')
profile = pd.DataFrame({'dtype':data.dtypes.astype(str), 'missing':data.isna().sum(), 'missing_pct':data.isna().mean()*100, 'unique':data.nunique(), 'blank_strings':data.apply(lambda s: s.astype('string').str.strip().eq('').sum())})
profile.to_csv(OUT/'column_profile.csv')
summary = numeric.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
summary['skew'] = numeric.skew()
summary['zero_count'] = numeric.eq(0).sum()
summary['iqr_flag_count'] = ((numeric.lt(numeric.quantile(.25)-1.5*(numeric.quantile(.75)-numeric.quantile(.25)))) | numeric.gt(numeric.quantile(.75)+1.5*(numeric.quantile(.75)-numeric.quantile(.25)))).sum()
summary.to_csv(OUT/'numeric_summary.csv')
dedup = data.drop_duplicates()
tracks = data.groupby('track_id', dropna=False).agg(rows=('track_genre','size'),genres=('track_genre','nunique'))
tracks[tracks.rows>1].to_csv(OUT/'repeated_track_ids.csv')
conflicts = data.groupby('track_id')[features+['artists','album_name','track_name','explicit','key','mode','time_signature']].nunique(dropna=False)
conflicts = conflicts.gt(1).sum().rename('track_ids_with_multiple_values')
conflicts.to_csv(OUT/'track_value_conflicts.csv')
genres = data.groupby('track_genre').agg(rows=('track_id','size'),unique_tracks=('track_id','nunique'),popularity_mean=('popularity','mean'),popularity_median=('popularity','median'),duration_median_ms=('duration_ms','median'))
genres['rows_after_exact_dedup'] = dedup.groupby('track_genre').size()
genres.to_csv(OUT/'genre_summary.csv')
pearson = numeric.corr()
spearman = numeric.rank().corr()
pearson.to_csv(OUT/'pearson_correlations.csv')
spearman.to_csv(OUT/'spearman_correlations.csv')
checks = {}
for c in ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']:
    checks[c+'_outside_0_1'] = ~data[c].between(0,1)
checks['popularity_outside_0_100'] = ~data.popularity.between(0,100)
checks['key_outside_0_11'] = ~data.key.isin(range(12))
checks['mode_outside_0_1'] = ~data['mode'].isin([0,1])
checks['duration_nonpositive'] = data.duration_ms.le(0)
checks['duration_over_20_minutes_review'] = data.duration_ms.gt(1200000)
checks['tempo_nonpositive_review'] = data.tempo.le(0)
checks['time_signature_zero_review'] = data.time_signature.eq(0)
checks['time_signature_outside_0_to_7'] = ~data.time_signature.isin(range(8))
checks['numeric_nonfinite'] = pd.Series(~np.isfinite(numeric.to_numpy()).all(axis=1), index=data.index)
flags = pd.DataFrame(checks)
flag_counts = flags.sum().rename('rows')
flag_counts.to_csv(OUT/'quality_checks.csv')
flagged = data.loc[flags.any(axis=1)].copy()
flagged['review_reasons'] = flags.loc[flags.any(axis=1)].apply(lambda r: '; '.join(r.index[r]),axis=1)
flagged.to_csv(OUT/'rows_to_review.csv',index_label='source_row_zero_based')
data.loc[data.isna().any(axis=1)].to_csv(OUT/'missing_rows.csv',index_label='source_row_zero_based')
data.loc[data.duplicated(keep=False)].to_csv(OUT/'duplicate_rows.csv',index_label='source_row_zero_based')
# Deterministic split simulation for measuring overlap, not a training split.
rng = np.random.default_rng(42)
order = rng.permutation(len(data)); cut = int(.8*len(data))
train, test = data.iloc[order[:cut]], data.iloc[order[cut:]]
overlap = int(test.track_id.isin(set(train.track_id)).sum())
artist_overlap = int(test.artists.isin(set(train.artists.dropna())).sum())
pairs = pearson.where(np.triu(np.ones(pearson.shape),k=1).astype(bool)).stack().rename('pearson_r').reset_index()
pairs.columns=['feature_1','feature_2','pearson_r']
pairs = pairs.sort_values('pearson_r',key=abs,ascending=False)
pairs.to_csv(OUT/'correlation_pairs.csv',index=False)
facts = {'rows':len(data),'columns_in_file':len(df.columns),'unique_tracks':data.track_id.nunique(),'exact_duplicates_excluding_index':int(data.duplicated().sum()),'repeated_track_ids':int(tracks.rows.gt(1).sum()),'multi_genre_track_ids':int(tracks.genres.gt(1).sum()),'rows_with_missing_values':int(data.isna().any(axis=1).sum()),'random_split_test_rows':len(test),'random_split_test_rows_with_train_track_id':overlap,'random_split_test_rows_with_train_artist_string':artist_overlap,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
(OUT/'metrics.json').write_text(json.dumps(facts,indent=2),encoding='utf-8')

def table(x):
    return x.to_html(border=0,float_format=lambda v:f'{v:,.3f}',classes='table')

def histogram(column, bins=30):
    vals=data[column].dropna().to_numpy()
    counts,edges=np.histogram(vals,bins=bins)
    bars=''.join(f'<rect x="{55+i*490/bins:.1f}" y="{190-v/max(counts)*150:.1f}" width="{490/bins-1:.1f}" height="{v/max(counts)*150:.1f}" fill="#167d9a"><title>{edges[i]:.3g}–{edges[i+1]:.3g}: {v:,} rows</title></rect>' for i,v in enumerate(counts))
    return f'<div class="chart"><h3>{column}</h3><svg viewBox="0 0 580 235" role="img" aria-label="Histogram of {column}"><text x="5" y="25">Rows (max bin {max(counts):,})</text>{bars}<text x="55" y="215">{edges[0]:.4g}</text><text x="480" y="215">{edges[-1]:.4g}</text></svg></div>'

def heatmap():
    corr=data[features].corr(); blocks=[]
    for i,a in enumerate(features):
        blocks.append(f'<text x="155" y="{175+i*39}" text-anchor="end">{a}</text>')
        blocks.append(f'<text transform="translate({180+i*49},145) rotate(-55)">{a}</text>')
        for j,b in enumerate(features):
            v=corr.loc[a,b]; color=f'rgb({int(245-abs(v)*190)},{int(245-abs(v)*100)},{int(245-abs(v)*40)})' if v>=0 else f'rgb(220,{int(245-abs(v)*140)},{int(245-abs(v)*170)})'
            blocks.append(f'<rect x="{165+j*49}" y="{150+i*39}" width="48" height="38" fill="{color}"/><text x="{189+j*49}" y="{175+i*39}" text-anchor="middle" fill="{"white" if v>.65 else "#172738"}">{v:.2f}</text>')
    return '<svg viewBox="0 0 730 600" role="img" aria-label="Pearson correlation matrix">'+''.join(blocks)+'</svg>'

recommendations = '''<ol>
<li><b>Define the prediction target and unit.</b> For genre, repeated IDs with multiple labels require a multilabel formulation or a justified single-label policy. Arbitrarily keeping the first genre discards valid information. For popularity, inspect conflicting values per ID before selecting or aggregating observations; no collection timestamp is available.</li>
<li><b>Remove the saved index from predictors.</b> Rows are arranged in genre blocks, making the index a target proxy for genre classification. Exclude track_id from predictors and retain it for grouping. Metadata such as artists and album names can encourage memorization; establish an audio-only baseline first.</li>
<li><b>Remove exact duplicates after excluding the index.</b> Recalculate label balance afterward. Keep separate legitimate genre memberships until the target policy is decided.</li>
<li><b>Split by track ID before fitting preprocessing.</b> Verify zero ID overlap across train/validation/test. For unseen-artist performance, split by artist groups, accounting for collaborators across semicolon-separated names. Group constraints may prevent exact class balance. Do not take contiguous slices of this genre-ordered file.</li>
<li><b>Review flagged records.</b> Investigate zero duration and tempo/time-signature zeros. Do not remove long tracks or IQR outliers automatically: they can represent legitimate musical styles. Zeros in instrumentalness or popularity should not automatically be treated as missing.</li>
<li><b>Fit transformations only on training data.</b> Impute missing metadata if used, encode key/mode/time signature appropriately rather than assuming a linear ordering, consider log1p duration for skew, and scale continuous features for distance-based or linear models. Tree models generally do not require scaling. Fit clipping thresholds on training data only.</li>
<li><b>Evaluate against simple baselines.</b> For genre use macro-F1, balanced accuracy, per-class recall, and a confusion matrix (or multilabel metrics if applicable). For popularity use MAE, RMSE, R² and a training-median baseline, with errors by genre and popularity band.</li>
<li><b>Reserve a final holdout before further model-directed exploration.</b> This EDA uses the whole supplied file and is descriptive. The exactly balanced genre sampling does not establish real-world genre prevalence; source provenance, dates and intended deployment population were not provided.</li></ol>'''
report = f'''<!doctype html><html><head><meta charset="utf-8"><title>Spotify dataset EDA</title><style>
body{{font:16px/1.6 system-ui,sans-serif;color:#172738;background:#f5f7fa;margin:0}}main{{max-width:1150px;margin:auto;background:white;padding:36px}}h1,h2,h3{{line-height:1.2}}h2{{margin-top:38px}}.cards,.charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}.card{{background:#eaf4f7;padding:20px;border-radius:8px}}.card strong{{font-size:30px;display:block}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px 10px;border-bottom:1px solid #dde4ea;text-align:right;white-space:nowrap}}th{{background:#edf2f6}}svg{{width:100%;height:auto;font:12px system-ui}}li{{margin-bottom:12px}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.cards,.charts{{grid-template-columns:1fr}}main{{padding:18px}}}}</style></head><body><main>
<h1>Spotify dataset: exploratory analysis</h1><p>Source: dataset.csv. Full-file descriptive analysis before model training. No model was trained and the source file was not changed. No target was specified.</p>
<div class="cards"><div class="card"><strong>{len(data):,}</strong>rows · {len(df.columns)} source columns</div><div class="card"><strong>{data.track_id.nunique():,}</strong>unique tracks · {len(genres)} genres</div></div>
<h2>Findings that affect training</h2><ul>
<li>All {len(genres)} genres have exactly 1,000 rows. After exact deduplication, per-genre counts range from {genres.rows_after_exact_dedup.min()} to {genres.rows_after_exact_dedup.max()}.</li>
<li>{facts['exact_duplicates_excluding_index']:,} redundant rows are exact duplicates when the saved index is excluded. {facts['repeated_track_ids']:,} IDs repeat, and {facts['multi_genre_track_ids']:,} IDs have multiple genre labels.</li>
<li>In a seeded random 80/20 row split, {overlap:,}/{len(test):,} test rows ({overlap/len(test):.1%}) share a track ID with training. This demonstrates leakage risk, not model performance.</li>
<li>{facts['rows_with_missing_values']} row(s) contain missing fields; numeric columns have {int(numeric.isna().sum().sum())} missing cells.</li>
<li>Popularity has mean {data.popularity.mean():.2f}, median {data.popularity.median():.0f}, and {data.popularity.eq(0).sum():,} zeros ({data.popularity.eq(0).mean():.1%}). Median duration is {data.duration_ms.median()/60000:.2f} minutes; maximum is {data.duration_ms.max()/60000:.2f} minutes.</li>
<li>There are {int(data.track_genre.ne(data.track_genre.shift()).sum())} contiguous genre runs. The saved index is sequential: {bool(np.array_equal(df['Unnamed: 0'].values,np.arange(len(df))))}. Its ordering exposes genre membership.</li></ul>
<h2>Schema and missingness</h2><div class="scroll">{table(profile)}</div>
<h2>Validity and unusual records</h2><p>Range checks reflect the apparent feature scales, not a supplied data dictionary. Review flags are not automatic deletion rules. IQR flags in the numeric summary identify distribution tails, not proven errors.</p>{table(flag_counts.to_frame())}
<h2>Numeric distributions</h2><p>Histograms use full observed ranges and 30 equal-width bins. Hover over bars for bin counts. Long tails can compress the bulk of duration values; quantiles are provided below.</p><div class="charts">{''.join(histogram(c) for c in features)}</div>
<h2>Summary statistics</h2><div class="scroll">{table(summary)}</div>
<h2>Feature relationships</h2><p>Pearson correlations are descriptive linear associations across rows, which weight repeated tracks multiple times. They do not establish causation or predictive performance. Discrete key, mode and time signature need separate encoding decisions.</p>{heatmap()}<h3>Strongest numeric pairs</h3>{table(pairs.head(10))}
<h3>Popularity associations</h3>{table(pd.DataFrame({'Pearson':pearson.popularity,'Spearman':spearman.popularity}).drop('popularity').sort_values('Spearman',key=abs,ascending=False))}
<h2>Repeated-track consistency</h2><p>Number of track IDs with more than one value in each field. Genre differences are counted separately above. Conflicts require a target-specific decision before collapsing tracks.</p>{table(conflicts.to_frame())}
<h2>Genre comparisons</h2><p>Descriptive row-weighted means. The following are the 10 highest and 10 lowest mean-popularity genres; all genres are included in genre_summary.csv.</p>{table(pd.concat([genres.sort_values('popularity_mean').tail(10).iloc[::-1],genres.sort_values('popularity_mean').head(10)]))}
<h2>Preparation and evaluation plan</h2>{recommendations}
<h2>Reproducibility</h2><p>Run <code>python eda.py</code> with pandas and numpy installed. The random overlap audit uses NumPy default_rng seed 42. Audit CSVs and metrics.json are saved alongside this report. File SHA-256: <code>{facts['source_sha256']}</code>.</p>
</main></body></html>'''
(OUT/'eda_report.html').write_text(report,encoding='utf-8')
print(json.dumps(facts,indent=2))
print(flag_counts.to_string())
print('Strongest pairs:\n',pairs.head(6).to_string(index=False))
print('Saved report:',OUT/'eda_report.html')
