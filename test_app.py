import unittest
from streamlit.testing.v1 import AppTest

class WebsiteTests(unittest.TestCase):
    def test_search_generate_and_reset(self):
        app = AppTest.from_file("app.py",default_timeout=60).run()
        self.assertEqual(len(app.exception),0)
        app.text_input[0].set_value("Comedy Gen Hoshino")
        app.button[0].click().run()
        self.assertEqual(len(app.selectbox),1)
        app.selectbox[0].select("5SuOikwiRyPMVoIQDJUgSV").run()
        app.button[1].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.session_state["playlist"]),20)
        seed = app.session_state["selected_song"]
        self.assertNotIn(seed,[r["track_id"] for r in app.session_state["playlist"]])
        app.toggle[0].set_value(True).run()
        app.button[1].click().run()
        self.assertFalse(any(r["explicit"] for r in app.session_state["playlist"]))
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

if __name__ == "__main__":
    unittest.main()





