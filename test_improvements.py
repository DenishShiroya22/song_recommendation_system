import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlparse, parse_qs
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from recommender import SongRecommender
from feedback_store import FeedbackStore
from spotify_client import (OAuthBroker, SpotifyClient, SpotifyError, UncertainCreation,
                            challenge, validate_settings, SCOPE)

def response(status=200, data=None, headers=None):
    item = Mock()
    item.status_code = status
    item.content = b"json"
    item.headers = headers or {}
    item.json.return_value = data or {}
    return item

class WeightedSearchTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame({"track_id":list("abcd"),
            "track_name":["Shape of You","Same sound","Raabta","NDA"],
            "artists":["Ed Sheeran","Other","Arijit Singh","Billie Eilish"],
            "track_genre":["pop","rock","pop","pop"],
            "spotify_url":["https://open.spotify.com/track/"+s for s in "abcd"],
            "explicit":[False,True,False,False]})
        self.engine = SongRecommender(self.catalog,
            sparse.csr_matrix([[1,0,1,0],[1,0,0,1],[0,1,1,0],[-1,0,1,0]],dtype=np.float32),2)

    def test_typo_search_and_exact_precedence(self):
        self.assertEqual(self.engine.search("shpe of you",1)[0]["track_id"],"a")
        self.assertEqual(self.engine.search("raabtaa",1)[0]["track_id"],"c")
        self.assertEqual(self.engine.search("arjit singh",1)[0]["track_id"],"c")
        self.assertEqual(self.engine.search("raabta",1)[0]["search_match"],"exact")
        self.assertEqual(self.engine.search("zzzzunfindablezzzz"),[])

    def test_weights_change_ranking_and_have_expected_scores(self):
        audio = self.engine.recommend("a",3,audio_weight=1)
        genre = self.engine.recommend("a",3,audio_weight=0)
        mixed = self.engine.recommend("a",3,audio_weight=.7)
        self.assertEqual(audio[0]["track_id"],"b")
        self.assertEqual(genre[0]["track_id"],"c")
        self.assertEqual([r["track_id"] for r in mixed],["b","c","d"])
        np.testing.assert_allclose([r["similarity"] for r in mixed],[.7,.65,.3],atol=1e-6)
        self.assertEqual(self.engine.recommend("a",3,True,audio_weight=1)[0]["track_id"],"c")
        for value in (-.1,1.1,float("nan"),True):
            with self.assertRaises(ValueError):
                self.engine.recommend("a",audio_weight=value)

    def test_weighted_save_load_preserves_block_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"test.joblib"
            self.engine.save(path)
            loaded=SongRecommender.load(path)
            self.assertEqual(loaded.recommend("a",3,audio_weight=.7),
                             self.engine.recommend("a",3,audio_weight=.7))

class FeedbackTests(unittest.TestCase):
    def test_persistence_upsert_clear_and_session_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"feedback.sqlite3"
            store=FeedbackStore(sqlite_path=path)
            rows=[{"track_id":"track","audio_similarity":.8,"genre_similarity":.5,
                   "similarity":.71,"ranking_version":"weighted_cosine_v1"}]
            request=store.record_impressions("session","seed",rows,.7,"request")
            store.record_impressions("session","seed",rows,.7,"request")
            store.vote("session",request,"track",1)
            store.vote("session",request,"track",0)
            reloaded=FeedbackStore(sqlite_path=path)
            report=reloaded.summary()[0]
            self.assertEqual((report["impressions"],report["ratings"],report["dislikes"]),(1,1,1))
            with self.assertRaises(ValueError):
                store.vote("attacker",request,"track",1)
            with self.assertRaises(ValueError):
                store.vote("session",request,"not-shown",1)
            with self.assertRaises(ValueError):
                store.vote("session",request,"track",5)
            store.vote("session",request,"track",None)
            self.assertIsNone(store.summary()[0]["like_rate_among_rated"])

class SpotifyTests(unittest.TestCase):
    def token_http(self):
        http=Mock()
        http.post.return_value=response(data={"access_token":"secret","refresh_token":"refresh",
            "expires_in":3600,"scope":SCOPE})
        return http

    def test_pkce_rfc_vector(self):
        self.assertEqual(challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
                         "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")

    def test_oauth_cross_tab_claim_single_use_and_wrong_state(self):
        broker=OAuthBroker()
        login=broker.begin("client","https://example.com/")
        params=parse_qs(urlparse(login["url"]).query)
        self.assertEqual(params["scope"],[SCOPE])
        self.assertNotIn("client_secret",params)
        self.assertIsNone(broker.claim(login["state"],login["claim_secret"]))
        http=self.token_http()
        with self.assertRaises(SpotifyError):
            broker.complete("forged","code",http=http)
        self.assertFalse(http.post.called)
        broker.complete(login["state"],"code",http=http)
        with self.assertRaises(SpotifyError):
            broker.claim(login["state"],"another-session")
        self.assertEqual(broker.claim(login["state"],login["claim_secret"])["token"]["access_token"],"secret")
        with self.assertRaises(SpotifyError):
            broker.complete(login["state"],"code",http=http)

    def test_login_expiry_denial_and_scope(self):
        clock=[0]
        broker=OAuthBroker(clock=lambda:clock[0])
        login=broker.begin("client","https://example.com/")
        clock[0]=601
        with self.assertRaises(SpotifyError):
            broker.claim(login["state"],login["claim_secret"])
        login=broker.begin("client","https://example.com/")
        http=self.token_http()
        broker.complete(login["state"],"",error="access_denied",http=http)
        self.assertIn("error",broker.claim(login["state"],login["claim_secret"]))
        self.assertFalse(http.post.called)
        login=broker.begin("client","https://example.com/")
        http.post.return_value=response(data={"access_token":"secret","expires_in":60,"scope":""})
        broker.complete(login["state"],"code",http=http)
        self.assertIn("error",broker.claim(login["state"],login["claim_secret"]))
        for uri in ("http://localhost:8501","http://example.com/","https://example.com/?a=b"):
            with self.assertRaises(SpotifyError):
                validate_settings("client",uri)

    def test_playlist_endpoints_order_and_refresh(self):
        http=self.token_http()
        http.request.side_effect=[response(201,{"id":"p"*22}),response(200,{"snapshot_id":"snap"})]
        token={"access_token":"expired","refresh_token":"refresh","expires_at":0}
        client=SpotifyClient("client",token,http)
        playlist_id=client.create_private_playlist("My songs")
        url=client.fill_new_playlist(playlist_id,["a"*22,"b"*22])
        self.assertTrue(http.post.called)
        calls=http.request.call_args_list
        self.assertEqual(calls[0].args,("POST","https://api.spotify.com/v1/me/playlists"))
        self.assertFalse(calls[0].kwargs["json"]["public"])
        self.assertEqual(calls[1].args,("PUT","https://api.spotify.com/v1/playlists/"+"p"*22+"/items"))
        self.assertEqual(calls[1].kwargs["json"]["uris"],["spotify:track:"+"a"*22,"spotify:track:"+"b"*22])
        self.assertEqual(url,"https://open.spotify.com/playlist/"+"p"*22)

    def test_lost_create_reply_is_never_retried(self):
        http=Mock()
        http.request.side_effect=requests.Timeout()
        client=SpotifyClient("client",{"access_token":"secret","expires_at":time.time()+1000},http)
        with self.assertRaises(UncertainCreation):
            client.create_private_playlist("Name")
        self.assertEqual(http.request.call_count,1)

    def test_denied_and_rate_limits_are_safe_errors(self):
        for status in (403,429):
            http=Mock()
            http.request.return_value=response(status,headers={"Retry-After":"12"})
            client=SpotifyClient("client",{"access_token":"secret","expires_at":time.time()+1000},http)
            with self.assertRaises(SpotifyError) as caught:
                client.create_private_playlist("Name")
            self.assertNotIn("secret",str(caught.exception))
            self.assertEqual(http.request.call_count,1)

if __name__=="__main__":
    unittest.main()

