# -------------------------------------------------------------------------------------------------------
# base class for cur pair scrapers from stockexshadow. Meant for educational purposes only!
# run: python3.7 -m scraper.investing.curpair.base
# -------------------------------------------------------------------------------------------------------
from scraper.web_scraper import WebScraper
from loguru import logger

class AbstractCurPairScraper(WebScraper):

    # ======================================================================================================
    # beg: constructor
    # ======================================================================================================
    def __init__(self, base, second, 
        at={
            'cur1': ['#quotes_summary_current_data', '.inlineblock', '.top'],
            'cur2': ['#quotes_summary_secondary_data'],
            'time': ['#quotes_summary_current_data', '.lighterGrayFont', '.bold'],
        }):
            super(AbstractCurPairScraper, self).__init__()
            
            self.at = at
            self.uri = f'https://www.investing.com/currencies/{base.lower()}-{second.lower()}-technical'
            self.init_preproc(at)
    # ======================================================================================================
    # end: constructor
    # ======================================================================================================
    

    # ======================================================================================================
    # beg: preprocess string from recursive find
    # ======================================================================================================
    def init_preproc(self, at):
        # Add all funcs here
        PREPOCESSORS = {
            'cur1': lambda s: s.split(),
            'cur2': lambda s: [line.split(": ")[-1] for line in s.split('\n')]
        }
        self.__add_prepoc(PREPOCESSORS)

    def __add_prepoc(self, PREPOCESSORS):
        """ to be used in `get_tckr` """
        self.__PREPROC = {}
        for k, _ in self.at.items():
            self.__PREPROC[k] = (lambda s: s) if k not in PREPOCESSORS else PREPOCESSORS[k]
    # ======================================================================================================
    # end: preprocess string from recursive find
    # ======================================================================================================


    # ======================================================================================================
    # beg: realtime ticker values (with some delay in seconds)
    # ======================================================================================================
    def get_tckr(self):
        """ reload resonse and send recent value"""
        self.goto(self.uri)
        ret = {}
        for k, loc in self.at.items():
            ret[k] = \
                self.__PREPROC[k](
                    self.recursive_find(loc))
        return ret
    # ======================================================================================================
    # end: realtime ticker values (with some delay in seconds)
    # ======================================================================================================



if __name__ == '__main__':
    
    eurusd = AbstractCurPairScraper('eur', 'usd')
    usdjpy = AbstractCurPairScraper('usd', 'jpy')
    usdbtc = AbstractCurPairScraper('usd', 'btc')
    
    while True:
    
        print(f'[EUR/USD] {eurusd.get_tckr()}')
        print(f'[USD/JPY] {usdjpy.get_tckr()}')
        print(f'[BTC/USD] {usdbtc.get_tckr()}')