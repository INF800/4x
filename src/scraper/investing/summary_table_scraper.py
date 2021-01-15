from scraper.web_scraper import WebScraper

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
                'ratio': lst[i+1],
                'Moving Avg': lst[i+3:i+7],
                'Indicators': lst[i+8:i+12],
                'Summary': lst[i+13:i+17],
            }
        return pairs_data


if __name__ == '__main__':
    
    from pprint import pprint
    uri = 'https://www.investing.com/technical/technical-summary'

    scraper = SummaryTableScraper(uri=uri, class_name='technicalSummaryTbl')
    data = scraper.get_pairs_info()

    pprint(data['EUR/USD'])
