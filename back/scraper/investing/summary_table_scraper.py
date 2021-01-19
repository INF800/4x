from scraper.web_scraper import WebScraper

uri = 'https://www.investing.com/technical/technical-summary'



class SummaryTableScraper(WebScraper):

    def __init__(self, uri, class_name):
        super(SummaryTableScraper, self).__init__()
        self.goto(uri)
        self.n_table_pairs = 12
        self.table_class_name = class_name
        self.technical_summary = self.__get_technical_summary()

    def __get_technical_summary(self):
        return self.find('.'+self.table_class_name, first=True).text.split('\n')[6:]

    def get_pairs_info(self):
        """ 
        returns pairs data with keys as cur pairs and
        values as dicts with keys - ratio ... summary
        """
        summary_list = self.__get_technical_summary()
        pairs_data = {}
        tot_pairs = len(summary_list)//self.n_table_pairs
        for i in range(0, len(summary_list), tot_pairs):
            pairs_data[summary_list[i]] = {
                'Pair'       : summary_list[i],
                'Ratio'      : summary_list[i+1],
                'MovingAvg'  : summary_list[i+3:i+7],
                'Indicators' : summary_list[i+8:i+12],
                'Summary'    : summary_list[i+13:i+17],
            }
        return pairs_data

def proc_pair_info(pair_info):
    """
    return true if all are either `Strong Buy` OR `Strong Sell`
    """
    if (len(set(pair_info['Summary'])) == 1) and (pair_info['Summary'][0][:6] == 'Strong'):
        print(f"[TRUE] \t {pair_info['Pair']} \t : {pair_info['Summary']}")
        return True
    print(f"[FALSE] \t {pair_info['Pair']} \t : {pair_info['Summary']}")
    return False


class PairScores:
    """
    Simple scores based on frequency [0,100]
    """
    def __init__(self):
        self.scores = {}
        for pair in ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'EUR/GBP', 'USD/CAD', 'NZD/USD', 'EUR/JPY', 'EUR/CHF', 'GBP/JPY', 'GBP/CHF']:
            self.scores[pair] = 0

    def increment(self, cur_strong_pair):
        self.scores[cur_strong_pair] = min(100, self.scores[cur_strong_pair]+1) 
    
    def decrement(self, cur_weak_pair):
        self.scores[cur_weak_pair] = max(0, self.scores[cur_weak_pair]-1) 





import os, time
from pprint import pprint

if __name__ == '__main__':

    scraper = SummaryTableScraper(uri=uri, class_name='technicalSummaryTbl')
    data = scraper.get_pairs_info()
    pair_scores = PairScores()

    while True:

        in_ = input('\nenter pair : ')
        
        scraper.goto(uri)
        data = scraper.get_pairs_info()

        for _, pair_info in data.items():
            
            if pair_info['Pair'] == in_:
                pprint(pair_info['Summary'])

    '''
    scraper = SummaryTableScraper(uri=uri, class_name='technicalSummaryTbl')
    data = scraper.get_pairs_info()
    pair_scores = PairScores()


    while True:

        scraper.goto(uri)
        data = scraper.get_pairs_info()

        for _, pair_info in data.items():
            
            strong = proc_pair_info(pair_info)

            if strong: 
                pair_scores.increment(pair_info['Pair']) 
            elif not strong: 
                pair_scores.decrement(pair_info['Pair']) 

        print(pair_scores.scores)
        print("="*200)
        time.sleep(3)
    '''