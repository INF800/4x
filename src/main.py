import requests
from requests_html import HTMLSession
 

class WebScraper:

    def __init__(self):
        self.sess = HTMLSession()
        self.cur_response = None

    def goto(self, uri):
        """ mutates response """
        try:
            self.cur_response = \
                self.sess.get(uri)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] {e}")

    def get_response(self):
        return self.cur_response
    


if __name__ == '__main__':

    uri = 'https://www.investing.com/technical/technical-summary'
    
    scraper = WebScraper()
    scraper.goto(uri)
    print(scraper.get_response())

