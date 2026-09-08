# Configure search, feedback, and Spotify playlist saving

Deployed app: https://songrecommendationsystem-wldmstqlaa4q5g4zzxtyg7.streamlit.app/

## 1. Fuzzy song search

Exact titles appear first, followed by substring matches. Within each group, higher dataset popularity comes first, with title, artist, and track ID breaking ties. Unknown popularity sorts last. This uses the dataset snapshot, not live listening counts. RapidFuzz fills remaining places with similar spellings for queries of at least three characters, ordered by spelling score and then popularity. A weighted score of at least 78 plus a full-string score of at least 50 avoids matching very short substrings inside nonsense queries. Examples: "shpe of you", "raabtaa", "arjit singh".

Users still explicitly choose a recording. Search never treats a spelling match as proof of the intended artist.

## 3. Separate sound and genre matching

The website generates a fixed 15-song playlist using 70% sound and 30% genre. Playlist settings are not shown to users; explicit songs remain eligible. The Python API still supports custom weights for evaluation.

The first eight columns of the existing matrix are standardized audio. Each audio vector and genre vector is normalized independently. For each candidate:

    sound = (cosine(audio_seed, audio_candidate) + 1) / 2
    genre = cosine(genre_seed, genre_candidate)
    score = audio_weight * sound + (1 - audio_weight) * genre

Sound is mapped to [0,1] because standardized features may have negative cosine similarities. Genre similarity is already nonnegative. Scores are ranking signals, not probabilities or accuracy percentages. A zero-norm audio block has cosine 0 and therefore sound score 0.5. Self matches and optionally explicit songs are filtered before ranking. Ties use track IDs for deterministic order.

The matrix and its row mapping remain unchanged. The Python API supports recommend(..., audio_weight=0.7). Omitting audio_weight keeps the earlier combined-cosine baseline for comparisons. No claimed relevance improvement or optimal weight has been established yet.

## 5. Recommendation feedback

Thumbs up/down save one current vote per recommended track per generated playlist. Changing or clearing a vote updates it rather than counting twice. The app records the shown rank, seed ID, weight, audio/genre scores, ranking version, and UTC timestamp. A random session ID isolates ratings without saving a Spotify user ID, email, or token.

Default storage is data/feedback.sqlite3. It works locally, but Streamlit Cloud's local disk is not durable across restarts/redeploys. For durable production feedback, provision a PostgreSQL database and add its server connection URL to Streamlit Secrets. The app creates recommendation_impressions and recommendation_votes tables. Use a dedicated database account limited to this application's tables, with TLS (sslmode=require). The database is accessed server-side only; do not put its URL in GitHub.

    feedback_database_url = "postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require"

A configured database error does not silently fall back to a local database. Recommendations remain available and the UI explains when ratings cannot be stored.

Run python feedback_store.py from a trusted owner environment to print aggregated impressions, ratings, likes, dislikes, response coverage, session count, and like rate among rated tracks grouped by weight. It reads FEEDBACK_DATABASE_URL or local Streamlit Secrets. Individual ratings and this aggregate report are not exposed on the public app.

Evaluate adequate samples before changing defaults. Like rates from users choosing different weights are observational and subject to selection bias. Unrated songs are not dislikes. The current votes do not automatically retrain or personalize the model.

## 6. Spotify connection setup

This uses Spotify Authorization Code with PKCE. No Spotify client secret is needed.

1. Create/select an app in https://developer.spotify.com/dashboard.
2. Add this exact redirect URI, including the trailing slash:

   https://songrecommendationsystem-wldmstqlaa4q5g4zzxtyg7.streamlit.app/

3. In Streamlit Community Cloud, open your app's Settings > Secrets and add:

    spotify_client_id = "YOUR_SPOTIFY_CLIENT_ID"
    spotify_redirect_uri = "https://songrecommendationsystem-wldmstqlaa4q5g4zzxtyg7.streamlit.app/"

The example file .streamlit/secrets.example.toml contains these fields plus the optional feedback database. Never commit the real .streamlit/secrets.toml.

For local testing register http://127.0.0.1:8501/ separately, use that address in the browser, and set spotify_redirect_uri accordingly. Spotify does not accept localhost redirect URIs.

Spotify development mode currently requires the app owner to have Premium and limits access to five allowlisted users (existing larger allowlists may be grandfathered). Add testers in the Spotify dashboard's Users and Access area. A successful login alone does not establish API access; unlisted users may get 403. Broad public Spotify-account access requires Spotify's applicable approval. Catalog search and recommendations are independent of OAuth.

### User flow

Connect Spotify above the search form and continue in the new tab. After Spotify approval, the callback tab processes the authorization and tells the user to return to the original Songside tab. That tab claims the connection automatically and keeps the generated playlist. The user supplies a playlist name and clicks Create private Spotify playlist. Only the displayed recommendations are exported, in order; the seed is excluded.

Only playlist-modify-private is requested. The account connection stays in the original Streamlit session and is cleared by Disconnect or session expiry. PKCE verifiers, one-time random state, and a separate claim secret protect the cross-tab exchange. Pending logins expire after ten minutes; tokens are never written to local disk, URLs, logs, feedback storage, or GitHub. They are briefly held in a locked in-memory handoff and then claimed by the initiating session. Keep the original tab open. An app restart or multiple independent server replicas requires reconnecting; a shared secure OAuth store would be needed for a multi-replica deployment.

Spotify API calls use POST /v1/me/playlists and PUT /v1/playlists/{id}/items. PUT is used only on the new playlist created by the current export, allowing a failed track upload to be retried without duplicating songs. The app retains the created playlist ID. If creation itself times out, it does not automatically retry the POST: users must check their Spotify library before generating a new export. A normal rerun or repeated click after success shows the existing playlist link.

### Validation limits

Automated tests cover search misspellings, separate-weight math, state resets, real rating callbacks, durable SQLite writes and vote changes, session isolation, PKCE verification, one-time state, denial/expiry, refresh, modern playlist routes and ordering, rate limits, and interrupted exports. Spotify network calls are mocked in tests. End-to-end Spotify authorization and real playlist creation still require configuring the client ID and an allowlisted account. The PostgreSQL adapter is provided but needs a live database smoke test after configuration.

## Official references

- https://developer.spotify.com/documentation/web-api/tutorials/code-pkce-flow
- https://developer.spotify.com/documentation/web-api/concepts/redirect_uri
- https://developer.spotify.com/documentation/web-api/reference/create-playlist
- https://developer.spotify.com/documentation/web-api/reference/reorder-or-replace-playlists-items
- https://developer.spotify.com/documentation/web-api/concepts/quota-modes
- https://docs.streamlit.io/develop/api-reference/widgets/st.feedback
- https://rapidfuzz.github.io/RapidFuzz/Usage/process.html
