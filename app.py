"""Run: python -m streamlit run app.py"""
import html
import json
import uuid
from pathlib import Path
import streamlit as st
from recommender import SongRecommender, DEFAULT_MODEL
from feedback_store import FeedbackStore
from spotify_ui import setting, handle_callback, render_connection, render_export

st.set_page_config(page_title="Songside · Find your next song", page_icon="🎧", layout="wide")
handle_callback()

@st.cache_resource
def feedback_store(database_url):
    return FeedbackStore(database_url)

def record_vote(request_id, track_id, widget_key):
    try:
        feedback_store(setting("feedback_database_url")).vote(
            st.session_state["listener_session"],request_id,track_id,st.session_state[widget_key])
        st.session_state.pop("feedback_error",None)
    except Exception:
        st.session_state["feedback_error"] = "Your rating could not be saved. Please try again."

st.session_state.setdefault("listener_session",uuid.uuid4().hex)
st.markdown("""
<style>
.stApp {background: #101415;}
.block-container {max-width: 1100px; padding-top: 2.5rem;}
h1 {letter-spacing: -0.055em; font-size: 3.5rem !important;}
.eyebrow {color:#9ae9bb; letter-spacing:.18em; font-size:.75rem; font-weight:700;}
.intro {color:#afbdb7; font-size:1.12rem; max-width:620px; margin-bottom:2rem;}
.track-title {font-weight:650; font-size:1.05rem; margin-bottom:.2rem; overflow-wrap:anywhere;}
.track-artist {color:#aab8b0; font-size:.9rem; overflow-wrap:anywhere;}
.number {color:#8daa9a; font-size:1rem; padding-top:.3rem;}
.hero {background:linear-gradient(115deg,#243e33,#182624);border:1px solid #375345;border-radius:18px;padding:24px;margin:12px 0 20px;}
.small-label {color:#acd2bc;font-size:.75rem;letter-spacing:.12em;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the song catalog…")
def load_engine(model_timestamp, engine_timestamp):
    return SongRecommender.load()

st.markdown('<div class="eyebrow">SONGSIDE / MUSIC DISCOVERY</div>', unsafe_allow_html=True)
st.title("One song. More to love.")
st.markdown('<p class="intro">Start with a song you love. Find a fresh playlist with a similar sound, mood, and genre.</p>', unsafe_allow_html=True)

if not DEFAULT_MODEL.exists():
    st.error("The song catalog is not ready yet.")
    st.caption("Setup: run python train_recommender.py, then restart this app.")
    st.stop()
try:
    engine = load_engine(DEFAULT_MODEL.stat().st_mtime_ns, Path(__file__).with_name("recommender.py").stat().st_mtime_ns)
except Exception:
    st.error("The song catalog could not be loaded. Rebuild the model and restart the app.")
    st.stop()

count = 15
clean_only = False
audio_weight = 0.7
render_connection()

with st.form("song_search"):
    query = st.text_input("Song title or artist", placeholder="Try Comedy Gen Hoshino", max_chars=200)
    submitted = st.form_submit_button("Search songs", type="primary")
if submitted:
    st.session_state.pop("playlist", None)
    st.session_state.pop("playlist_signature", None)
    st.session_state.pop("selected_song", None)
    st.session_state.pop("preview_song", None)
    st.session_state["search_generation"] = st.session_state.get("search_generation", 0) + 1
    st.session_state["searched"] = True
    st.session_state["matches"] = engine.search(query,limit=30) if query.strip() else []
    st.session_state["empty_query"] = not query.strip()

matches = st.session_state.get("matches", [])
if not st.session_state.get("searched"):
    st.caption("Try a title, an artist, or both. You’ll choose the recording before we build your playlist.")
elif st.session_state.get("empty_query"):
    st.warning("Enter a song title or artist to search.")
elif not matches:
    st.info("No songs found. Try fewer words, check the spelling, or search by artist.")
else:
    lookup = {r["track_id"]:r for r in matches}
    selection_key = "selected_song_" + str(st.session_state.get("search_generation", 0))
    selected = st.selectbox("Choose a song", options=list(lookup), index=None, placeholder="Select the song and artist you want", key=selection_key,
        format_func=lambda key: lookup[key]["track_name"]+" — "+lookup[key]["artists"].replace(";", ", ")+" · "+lookup[key]["album_name"]+" · "+key[-6:])
    st.session_state["selected_song"] = selected
    if selected is None:
        st.info("Choose the artist and recording you want. Songs with the same title can be covers or remixes.")
        st.stop()
    seed = lookup[selected]
    st.markdown('<div class="hero"><div class="small-label">YOUR STARTING TRACK</div><h2>'+html.escape(seed["track_name"])+'</h2><div>'+html.escape(seed["artists"].replace(";", ", "))+'</div><p class="track-artist">'+html.escape(seed["track_genre"].replace(";", " · "))+'</p></div>',unsafe_allow_html=True)
    st.caption("Recording: " + seed["album_name"])
    st.link_button("Open this recording in Spotify ↗", seed["spotify_url"])
    signature = (selected,count,clean_only,audio_weight)
    if st.button("Generate playlist",type="primary"):
        with st.spinner("Finding your next favorites…"):
            st.session_state["playlist"] = engine.recommend(selected,k=count,exclude_explicit=clean_only,audio_weight=audio_weight)
            st.session_state["playlist_signature"] = signature
            request_id = uuid.uuid4().hex
            st.session_state["playlist_request"] = request_id
            try:
                feedback_store(setting("feedback_database_url")).record_impressions(
                    st.session_state["listener_session"],selected,st.session_state["playlist"],audio_weight,request_id)
                st.session_state["feedback_ready"] = True
                st.session_state.pop("feedback_error",None)
            except Exception:
                st.session_state["feedback_ready"] = False
                st.session_state["feedback_error"] = "Ratings are temporarily unavailable. You can still use your playlist."
    if st.session_state.get("playlist_signature") == signature:
        playlist = st.session_state.get("playlist",[])
        request_id = st.session_state.get("playlist_request", "legacy")
        st.divider()
        st.subheader("Your next listens")
        st.caption(f"{len(playlist)} songs inspired by "+seed["track_name"])
        st.caption("Rate each match to help evaluate recommendations. Ratings use a random browser-session ID, with no Spotify account details.")
        if not setting("feedback_database_url"):
            st.caption("Ratings are stored temporarily on this server and may reset when the app restarts.")
        if st.session_state.get("feedback_error"):
            st.warning(st.session_state["feedback_error"])
        if not playlist:
            st.info("No recommendations found. Choose another song.")
        for rank,track in enumerate(playlist,1):
            with st.container(border=True):
                number, details, link = st.columns([.4,5,1.8],vertical_alignment="center")
                number.markdown('<div class="number">'+str(rank).zfill(2)+'</div>',unsafe_allow_html=True)
                details.markdown('<div class="track-title">'+html.escape(track["track_name"])+'</div><div class="track-artist">'+html.escape(track["artists"].replace(";", ", "))+'</div>',unsafe_allow_html=True)
                reasons = ["Similar sound"]
                if track["shared_genres"]:
                    reasons.append("Shared genre: "+", ".join(track["shared_genres"][:3]))
                if track["same_artist"]:
                    reasons.append("Same artist")
                if str(track["explicit"]).lower()=="true":
                    reasons.append("Explicit")
                details.caption(" · ".join(reasons))
                if "audio_similarity" in track:
                    details.caption(f"Sound match {track['audio_similarity']:.2f} · Genre match {track['genre_similarity']:.2f}")
                if st.session_state.get("feedback_ready"):
                    with details:
                        vote_key = "vote_"+request_id+"_"+track["track_id"]
                        st.feedback("thumbs",key=vote_key,on_change=record_vote,
                                    args=(request_id,track["track_id"],vote_key))
                link.link_button("Open in Spotify ↗",track["spotify_url"])
        if playlist:
            preview_lookup = {track["track_id"]: track for track in playlist}
            preview_id = st.selectbox(
                "Preview an exact recommendation",
                options=list(preview_lookup),
                index=None,
                placeholder="Choose a song to play inside this page",
                key="preview_song_" + request_id,
                format_func=lambda key: preview_lookup[key]["track_name"] + " — " + preview_lookup[key]["artists"].replace(";", ", "),
            )
            if preview_id:
                st.session_state["preview_song"] = preview_id
                st.caption("Spotify track ID: " + preview_id)
                st.iframe(
                    "https://open.spotify.com/embed/track/" + preview_id + "?utm_source=generator",
                    height=160,
                )
                st.caption(
                    "If Spotify skips after opening its website, check the queue and shuffle setting. "
                    "Free mobile playback may choose a different song."
                )
            st.download_button("Download playlist links",
                data="\n".join(r["spotify_url"] for r in playlist),
                file_name="songside_playlist.txt",mime="text/plain")
            render_export(seed,playlist,request_id)
    elif "playlist" in st.session_state:
        st.caption("Your selection changed. Generate a new playlist to update the results.")








