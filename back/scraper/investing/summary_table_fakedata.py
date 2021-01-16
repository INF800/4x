import random

summary_table_fakedata = []


all_summs = [
    ['Strong Sell']*4,                      # 1
    ['Strong Buy']*4,                       # 2
    ['Strong Buy']*2 + ['Strong Sell']*2,   # 3
    ['Buy']*4,                              # 4
    ['Sell']*4,                             # 5
    ['Neutral']*4,                          # 6
]


for cntr in range(100):
    
    __temp = {}
    for pair in ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'EUR/GBP', 'USD/CAD', 'NZD/USD', 'EUR/JPY', 'EUR/CHF', 'GBP/JPY', 'GBP/CHF']:
        
        __temp[pair] = {
            'Pair': pair,
            'Summary': random.choice(all_summs)
        }

    summary_table_fakedata.append(__temp)
