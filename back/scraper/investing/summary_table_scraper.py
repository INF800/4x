from scraper.web_scraper import WebScraper

uri = 'https://www.investing.com/technical/technical-summary'



class SummaryTableScraper(WebScraper):

    def __init__(self, uri, class_name):
        super(SummaryTableScraper, self).__init__()
        self.goto(uri)
        self.n_table_pairs = 12
        self.technical_summary = self.find('.'+class_name, first=True)

    def get_pairs_info(self):
        """ 
        returns pairs data with keys as cur pairs and
        values as dicts with keys - ratio ... summary
        """
        lst = self.technical_summary.text.split('\n')[6:]
        pairs_data = {}
        for i in range(0, len(lst), int(len(lst)/self.n_table_pairs)):
            pairs_data[lst[i]] = {
                'Pair'       : lst[i],
                'Ratio'      : lst[i+1],
                'MovingAvg'  : lst[i+3:i+7],
                'Indicators' : lst[i+8:i+12],
                'Summary'    : lst[i+13:i+17],
            }
        return pairs_data



def proc_pair_info(pair_info):
    """
    return true if all are either `Strong Buy` OR `Strong Sell`
    """
    if (len(set(pair_info['Summary'])) == 1) and (pair_info['Summary'][0][:6] == 'Strong'):
        return True
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



if __name__ == '__main__':
    
    import os, time
    from pprint import pprint

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