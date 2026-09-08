"""Spotify PKCE login and private playlist export. No client secret or global token cache."""
import base64
import hashlib
import re
import secrets
import threading
import time
from urllib.parse import urlencode, urlparse
import requests

SCOPE = "playlist-modify-private"
API = "https://api.spotify.com/v1"
TOKEN_URL = "https://accounts.spotify.com/api/token"

class SpotifyError(Exception):
    pass

class UncertainCreation(SpotifyError):
    """A POST may have reached Spotify; never automatically repeat playlist creation."""
    pass

def validate_settings(client_id, redirect_uri):
    parsed = urlparse(redirect_uri)
    secure = parsed.scheme == "https" or (parsed.scheme == "http" and parsed.hostname in ("127.0.0.1","::1"))
    if not client_id or not parsed.netloc or not secure or parsed.query or parsed.fragment or parsed.username:
        raise SpotifyError("Configure a Spotify client ID and an exact HTTPS redirect URI (or local loopback IP).")

def challenge(verifier):
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")

def authorization_url(client_id, redirect_uri, state, verifier):
    validate_settings(client_id, redirect_uri)
    return "https://accounts.spotify.com/authorize?" + urlencode({
        "client_id":client_id, "response_type":"code", "redirect_uri":redirect_uri,
        "scope":SCOPE, "state":state, "code_challenge_method":"S256",
        "code_challenge":challenge(verifier), "show_dialog":"true"})

class OAuthBroker:
    """Short-lived cross-tab handoff. Original tab alone knows the separate claim secret."""
    def __init__(self, clock=time.time):
        self.pending = {}
        self.lock = threading.Lock()
        self.clock = clock

    def prune(self):
        now = self.clock()
        self.pending = {k:v for k,v in self.pending.items() if v["expires"] > now}

    def begin(self, client_id, redirect_uri):
        state, claim_secret, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(32), secrets.token_urlsafe(64)
        url = authorization_url(client_id, redirect_uri, state, verifier)
        with self.lock:
            self.prune()
            if len(self.pending) >= 1000:
                raise SpotifyError("Too many pending sign-ins. Please try again later.")
            self.pending[state] = {"claim_hash":hashlib.sha256(claim_secret.encode()).digest(),
                "verifier":verifier, "expires":self.clock()+600, "status":"pending",
                "client_id":client_id, "redirect_uri":redirect_uri}
        return {"state":state, "claim_secret":claim_secret, "url":url}

    def complete(self, state, code, error=None, http=requests):
        with self.lock:
            self.prune()
            entry = self.pending.get(state)
            if not entry or entry["status"] != "pending":
                raise SpotifyError("This sign-in link expired or was already used. Start again from the original tab.")
            entry["status"] = "exchanging"
        try:
            if error:
                raise SpotifyError("Spotify sign-in was cancelled or denied.")
            if not code or len(code) > 4096:
                raise SpotifyError("Spotify did not return a valid authorization code.")
            token = token_request({"grant_type":"authorization_code", "code":code,
                "redirect_uri":entry["redirect_uri"], "client_id":entry["client_id"],
                "code_verifier":entry["verifier"]}, http)
            if SCOPE not in token.get("scope", "").split():
                raise SpotifyError("Spotify did not grant private-playlist permission.")
            result = {"token":token}
        except SpotifyError as exc:
            result = {"error":str(exc)}
        with self.lock:
            if state in self.pending:
                entry.pop("verifier", None)
                entry["result"] = result
                entry["status"] = "ready"

    def claim(self, state, claim_secret):
        with self.lock:
            self.prune()
            entry = self.pending.get(state)
            if not entry:
                raise SpotifyError("Sign-in expired. Please connect Spotify again.")
            digest = hashlib.sha256(claim_secret.encode()).digest()
            if not secrets.compare_digest(digest, entry["claim_hash"]):
                raise SpotifyError("Sign-in does not belong to this browser session.")
            if entry["status"] != "ready":
                return None
            return self.pending.pop(state)["result"]

def token_request(data, http=requests):
    try:
        response = http.post(TOKEN_URL, data=data, timeout=15)
    except requests.RequestException:
        raise SpotifyError("Spotify sign-in could not finish. Please connect again.") from None
    if response.status_code != 200:
        raise SpotifyError("Spotify rejected the login or refresh request. Please reconnect.")
    try:
        token = response.json()
        if not token.get("access_token") or float(token.get("expires_in",0)) <= 0:
            raise ValueError()
        token["expires_at"] = time.time() + float(token["expires_in"])
    except (ValueError, TypeError, AttributeError):
        raise SpotifyError("Spotify returned an invalid login response.") from None
    return token

class SpotifyClient:
    def __init__(self, client_id, token, http=requests):
        self.client_id, self.token, self.http = client_id, token, http

    def refresh(self):
        if not self.token.get("refresh_token"):
            raise SpotifyError("Spotify connection expired. Please reconnect.")
        refreshed = token_request({"grant_type":"refresh_token","refresh_token":self.token["refresh_token"],
                                   "client_id":self.client_id}, self.http)
        self.token.update(refreshed)

    def request(self, method, path, body=None):
        if self.token.get("expires_at",0) < time.time()+30:
            self.refresh()
        for attempt in range(2):
            try:
                response = self.http.request(method, API+path,
                    headers={"Authorization":"Bearer "+self.token["access_token"]},
                    json=body, timeout=15)
            except requests.RequestException:
                if method == "POST":
                    raise UncertainCreation("Spotify may have created the playlist, but its reply was lost. Check your Spotify library before creating another.") from None
                raise SpotifyError("Spotify could not be reached. You can retry adding the songs.") from None
            if response.status_code == 401 and attempt == 0:
                self.refresh()
                continue
            if response.status_code == 429:
                wait = response.headers.get("Retry-After","a few")
                raise SpotifyError(f"Spotify reached a rate or quota limit. Try later (retry-after: {wait} seconds).")
            if response.status_code == 403:
                raise SpotifyError("Spotify denied access. Check the app's allowed users and playlist permissions.")
            if response.status_code >= 500 and method == "POST":
                raise UncertainCreation("Spotify returned a server error after playlist creation. Check your library before creating another.")
            if not 200 <= response.status_code < 300:
                raise SpotifyError(f"Spotify request failed (HTTP {response.status_code}).")
            try:
                return response.json() if response.content else {}
            except ValueError:
                if method == "POST":
                    raise UncertainCreation("Spotify may have created the playlist but returned an unreadable reply. Check your library.")
                raise SpotifyError("Spotify returned an unreadable response.") from None
        raise SpotifyError("Spotify connection expired. Reconnect.")

    def create_private_playlist(self, name):
        name = name.strip()
        if not 1 <= len(name) <= 100:
            raise ValueError("Playlist name must contain 1–100 characters.")
        result = self.request("POST","/me/playlists",
            {"name":name,"public":False,"description":"Songside recommendations based on your selected song."})
        playlist_id = result.get("id","")
        if not re.fullmatch(r"[A-Za-z0-9]{22}", playlist_id):
            raise UncertainCreation("Playlist creation returned no valid ID. Check your Spotify library.")
        return playlist_id

    def fill_new_playlist(self, playlist_id, track_ids):
        # Only called with the ID created by this export. PUT makes retry safe after a lost reply.
        if not re.fullmatch(r"[A-Za-z0-9]{22}",playlist_id):
            raise ValueError("Invalid playlist ID.")
        if not 1 <= len(track_ids) <= 100 or len(set(track_ids)) != len(track_ids):
            raise ValueError("Export requires 1–100 unique tracks.")
        if any(not re.fullmatch(r"[A-Za-z0-9]{22}",track) for track in track_ids):
            raise ValueError("Invalid track ID.")
        self.request("PUT",f"/playlists/{playlist_id}/items",
                     {"uris":["spotify:track:"+track for track in track_ids]})
        return "https://open.spotify.com/playlist/"+playlist_id

