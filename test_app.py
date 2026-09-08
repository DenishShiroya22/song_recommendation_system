import unittest
import time
from unittest.mock import patch
from spotify_client import SpotifyError, UncertainCreation
from streamlit.testing.v1 import AppTest

class WebsiteTests(unittest.TestCase):
    @staticmethod
    def click_label(app, label):
        next(button for button in app.button if button.label == label).click().run()

    def connected_playlist(self):
        app = AppTest.from_file("app.py",default_timeout=60)
        app.secrets["spotify_client_id"] = "test-client"
        app.session_state["spotify_token"] = {"access_token":"test", "expires_at":time.time()+1000}
        app.run()
        app.text_input[0].set_value("Comedy Gen Hoshino")
        self.click_label(app,"Search songs")
        app.selectbox[0].select("5SuOikwiRyPMVoIQDJUgSV").run()
        self.click_label(app,"Generate playlist")
        return app

    def test_spotify_export_explicit_click_and_retry_without_duplicate(self):
        with patch("spotify_ui.SpotifyClient") as client_class:
            client = client_class.return_value
            client.create_private_playlist.return_value = "p"*22
            client.fill_new_playlist.side_effect = [SpotifyError("temporary"),"https://open.spotify.com/playlist/"+"p"*22]
            app = self.connected_playlist()
            self.assertFalse(client.create_private_playlist.called)
            self.click_label(app,"Create private Spotify playlist")
            self.assertEqual(client.create_private_playlist.call_count,1)
            self.click_label(app,"Retry adding songs")
            self.assertEqual(client.create_private_playlist.call_count,1)
            self.assertEqual(client.fill_new_playlist.call_count,2)
            app.run()
            self.assertEqual(client.fill_new_playlist.call_count,2)
            self.assertEqual(len(app.exception),0)
            request_id = app.session_state["playlist_request"]
            self.assertEqual(app.session_state["spotify_exports"][request_id]["status"],"saved")

    def test_spotify_uncertain_creation_does_not_repeat(self):
        with patch("spotify_ui.SpotifyClient") as client_class:
            client_class.return_value.create_private_playlist.side_effect = UncertainCreation("Reply lost")
            app = self.connected_playlist()
            self.click_label(app,"Create private Spotify playlist")
            app.run()
            self.assertEqual(client_class.return_value.create_private_playlist.call_count,1)
            self.assertFalse(any(b.label == "Create private Spotify playlist" for b in app.button))
            self.assertEqual(len(app.exception),0)

    def test_search_generate_and_reset(self):
        app = AppTest.from_file("app.py",default_timeout=60).run()
        self.assertEqual(len(app.exception),0)
        app.text_input[0].set_value("Comedy Gen Hoshino")
        app.button[0].click().run()
        self.assertEqual(len(app.selectbox),1)
        app.selectbox[0].select("5SuOikwiRyPMVoIQDJUgSV").run()
        app.button[1].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.session_state["playlist"]),15)
        self.assertEqual(len(app.slider),0)
        self.assertEqual(len(app.toggle),0)
        seed = app.session_state["selected_song"]
        self.assertNotIn(seed,[r["track_id"] for r in app.session_state["playlist"]])
        preview_id = app.session_state["playlist"][0]["track_id"]
        app.selectbox[1].select(preview_id).run()
        self.assertEqual(app.session_state["preview_song"], preview_id)
        self.assertEqual(len(app.exception),0)
        app.text_input[0].set_value("zzzzunfindablezzzz")
        app.button[0].click().run()
        self.assertEqual(len(app.selectbox),0)
        self.assertEqual(len(app.info),1)
        self.assertEqual(len(app.exception),0)

    def test_shape_of_you_requires_recording_selection(self):
        app = AppTest.from_file("app.py",default_timeout=60).run()
        app.text_input[0].set_value("shape of you")
        app.button[0].click().run()
        self.assertIsNone(app.selectbox[0].value)
        self.assertEqual(len(app.button),1)
        app.selectbox[0].select("7qiZfU4dY1lWllzX7mPBI3").run()
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any("Ed Sheeran" in item.value for item in app.markdown))
        links = app.get("link_button")
        self.assertTrue(any(item.proto.url == "https://open.spotify.com/track/7qiZfU4dY1lWllzX7mPBI3" for item in links))

    def test_new_search_resets_previous_song_selection(self):
        app = AppTest.from_file("app.py",default_timeout=60).run()
        app.text_input[0].set_value("shape of you")
        app.button[0].click().run()
        app.selectbox[0].select("7qiZfU4dY1lWllzX7mPBI3").run()
        self.assertEqual(app.session_state["selected_song"], "7qiZfU4dY1lWllzX7mPBI3")
        app.text_input[0].set_value("raabta")
        app.button[0].click().run()
        self.assertIsNone(app.selectbox[0].value)
        self.assertIsNone(app.session_state["selected_song"])
        result_ids = [row["track_id"] for row in app.session_state["matches"]]
        self.assertIn("6FjbAnaPRPwiP3sciEYctO", result_ids)
        self.assertNotIn("7qiZfU4dY1lWllzX7mPBI3", result_ids)
        self.assertEqual(len(app.exception),0)
    def test_empty_search(self):
        app = AppTest.from_file("app.py",default_timeout=60).run()
        app.button[0].click().run()
        self.assertEqual(len(app.warning),1)
        self.assertEqual(len(app.exception),0)

    def test_typo_fixed_playlist_and_feedback(self):
        app = AppTest.from_file("app.py",default_timeout=60).run()
        app.text_input[0].set_value("shpe of you")
        app.button[0].click().run()
        self.assertIn("7qiZfU4dY1lWllzX7mPBI3",[r["track_id"] for r in app.session_state["matches"]])
        self.assertFalse(any("matches shown" in c.value or "Exact titles first" in c.value
                             or "Includes similar spellings" in c.value for c in app.caption))
        app.selectbox[0].select("7qiZfU4dY1lWllzX7mPBI3").run()
        app.button[1].click().run()
        first_request = app.session_state["playlist_request"]
        self.assertEqual(app.session_state["playlist"][0]["audio_weight"],.7)
        self.assertTrue(app.session_state["feedback_ready"])
        app.get("feedback")[0].set_value(1).run()
        self.assertEqual(len(app.exception),0)
        self.assertNotIn("feedback_error",app.session_state)
        app.button[1].click().run()
        self.assertNotEqual(first_request,app.session_state["playlist_request"])
        self.assertEqual(len(app.session_state["playlist"]),15)
        self.assertEqual(app.session_state["playlist"][0]["audio_weight"],.7)
        self.assertEqual(len(app.exception),0)

if __name__ == "__main__":
    unittest.main()





