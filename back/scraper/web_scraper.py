import requests
from requests_html import HTMLSession

class WebScraper:

    def __init__(self):
        self.sess = HTMLSession()
        self.cur_response = None

    def goto(self, uri):
        """ mutate `self.cur_response` """
        try:
            self.cur_response = \
                self.sess.get(uri)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {e}")

    def get_response(self):
        return self.cur_response

    def find(self, s, **kwargs):
        """finds html tags, .classes, #ids"""
        return self.cur_response.html.find(s, **kwargs)