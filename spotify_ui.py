"""Streamlit presentation for optional Spotify connection and private playlist export."""
import os
from datetime import timedelta
import streamlit as st
from spotify_client import OAuthBroker, SpotifyClient, SpotifyError, UncertainCreation

DEPLOYED_URL = "https://songrecommendationsystem-wldmstqlaa4q5g4zzxtyg7.streamlit.app/"

def setting(name, default=""):
    try:
        return str(st.secrets.get(name, os.environ.get(name.upper(), default)))
    except st.errors.StreamlitSecretNotFoundError:
        return os.environ.get(name.upper(), default)

@st.cache_resource
def broker():
    return OAuthBroker()

def handle_callback():
    if "code" not in st.query_params and "error" not in st.query_params:
        return
    state = st.query_params.get("state","")
    code = st.query_params.get("code","")
    error = st.query_params.get("error")
    for key in ("code","state","error","error_description"):
        if key in st.query_params:
            del st.query_params[key]
    st.title("Spotify connection")
    try:
        broker().complete(state, code, error)
        st.success("Sign-in processed. Return to your original Songside tab to continue.")
        st.caption("You can close this tab. Your playlist remains in the original tab.")
    except SpotifyError as exc:
        st.error(str(exc))
    st.stop()

@st.fragment(run_every=timedelta(seconds=2))
def render_connection():
    client_id = setting("spotify_client_id")
    if not client_id:
        return
    if st.session_state.get("spotify_token"):
        st.caption("Spotify connected for this browser session.")
        if st.button("Disconnect Spotify",key="disconnect_spotify"):
            st.session_state.pop("spotify_token",None)
            st.session_state.pop("spotify_login",None)
            st.session_state.pop("spotify_exports",None)
            st.rerun()
        return
    pending = st.session_state.get("spotify_login")
    if pending:
        try:
            result = broker().claim(pending["state"],pending["claim_secret"])
            if result:
                st.session_state.pop("spotify_login",None)
                if "error" in result:
                    st.session_state["spotify_login_error"] = result["error"]
                else:
                    st.session_state["spotify_token"] = result["token"]
                st.rerun()
        except SpotifyError as exc:
            st.session_state.pop("spotify_login",None)
            st.session_state["spotify_login_error"] = str(exc)
            pending = None
    if st.session_state.get("spotify_login_error"):
        st.warning(st.session_state["spotify_login_error"])
    if not pending:
        if st.button("Connect Spotify",key="connect_spotify"):
            try:
                pending = broker().begin(client_id,setting("spotify_redirect_uri",DEPLOYED_URL))
                st.session_state["spotify_login"] = pending
                st.session_state.pop("spotify_login_error",None)
            except SpotifyError as exc:
                st.warning(str(exc))
    if pending:
        st.link_button("Continue to Spotify",pending["url"])
        st.caption("Sign in in the new tab, then return here. The connection updates automatically.")

def render_export(seed, playlist, request_id):
    st.subheader("Save to Spotify")
    client_id = setting("spotify_client_id")
    if not client_id:
        st.caption("Spotify account connection is not enabled on this deployment yet. You can still open tracks or download their links.")
        return
    if not st.session_state.get("spotify_token"):
        st.caption("Connect Spotify in the sidebar, then return here to save these songs.")
        return
    name = st.text_input("Playlist name",value=("Songside - "+seed["track_name"])[:100],
                         max_chars=100,key="export_name_"+request_id)
    st.caption(f"Create a private playlist with these {len(playlist)} recommended songs, in the displayed order.")
    exports = st.session_state.setdefault("spotify_exports",{})
    export = exports.get(request_id,{})
    status = export.get("status")
    if export.get("error"):
        st.warning(export["error"])
    if status == "saved":
        st.link_button("Open your saved playlist",export["url"])
        return
    if status in ("uncertain","creating"):
        st.warning("Spotify may have created this playlist, but confirmation was lost. Check your Spotify library before trying again with a newly generated playlist.")
        return
    if status == "needs_tracks":
        st.link_button("Open the new playlist","https://open.spotify.com/playlist/"+export["id"])
        st.caption("The playlist was created, but its tracks still need to be added. Retrying replaces its contents with the displayed songs.")
    label = "Retry adding songs" if status == "needs_tracks" else "Create private Spotify playlist"
    if st.button(label,key="export_"+request_id,disabled=not name.strip()):
        client = SpotifyClient(client_id,st.session_state["spotify_token"])
        try:
            if status != "needs_tracks":
                export = {"status":"creating"}
                exports[request_id] = export
                playlist_id = client.create_private_playlist(name)
                export.update(id=playlist_id,status="needs_tracks")
            url = client.fill_new_playlist(export["id"],[r["track_id"] for r in playlist])
            export.update(status="saved",url=url)
            export.pop("error",None)
            st.rerun()
        except UncertainCreation as exc:
            export["status"] = "uncertain"
            export["error"] = str(exc)
            st.rerun()
        except (SpotifyError,ValueError) as exc:
            if export.get("status") == "creating":
                export["status"] = "retry"
            export["error"] = str(exc)
            st.rerun()
