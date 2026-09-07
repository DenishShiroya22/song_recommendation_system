import unittest
from streamlit.testing.v1 import AppTest

class WebsiteTests(unittest.TestCase):
    def test_search_generate_and_reset(self):
        app = AppTest.from_file("app.py",default_timeout=60).run()
        self.assertEqual(len(app.exception),0)
        app.text_input[0].set_value("Comedy Gen Hoshino")
        app.button[0].click().run()
        self.assertEqual(len(app.selectbox),1)
        app.button[1].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.session_state["playlist"]),20)
        seed = app.session_state["selected_song"]
        self.assertNotIn(seed,[r["track_id"] for r in app.session_state["playlist"]])
        app.toggle[0].set_value(True).run()
        app.button[1].click().run()
        self.assertFalse(any(r["explicit"] for r in app.session_state["playlist"]))
        app.text_input[0].set_value("zzzzunfindablezzzz")
        app.button[0].click().run()
        self.assertEqual(len(app.selectbox),0)
        self.assertEqual(len(app.info),1)
        self.assertEqual(len(app.exception),0)

    def test_empty_search(self):
        app = AppTest.from_file("app.py",default_timeout=60).run()
        app.button[0].click().run()
        self.assertEqual(len(app.warning),1)
        self.assertEqual(len(app.exception),0)

if __name__ == "__main__":
    unittest.main()

