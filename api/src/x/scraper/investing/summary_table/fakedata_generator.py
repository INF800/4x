import random


summary_table_fakedata = []
max_cntr = 30
all_summs = [
    ['Strong Sell']*4,                      # 0
    ['Strong Buy']*4,                       # 1
    ['Strong Buy']*2 + ['Strong Sell']*2,   # 2
    ['Buy']*4,                              # 3
    ['Sell']*4,                             # 4
    ['Neutral']*4,                          # 5
]


for cntr in range(max_cntr):
    
    __temp = {}
    for pair in ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'EUR/GBP', 'USD/CAD', 'NZD/USD', 'EUR/JPY', 'EUR/CHF', 'GBP/JPY', 'GBP/CHF']:
        
        __summ = all_summs[5] # Neutral

        if (pair == 'EUR/USD') and (cntr <10):
            __summ = all_summs[0] # strong sell
        elif (pair == 'USD/CAD') and (cntr <20):
            __summ = all_summs[1] # strong buy
        elif (pair == 'USD/JPY') and (cntr >20):
            __summ = all_summs[0] # strong sell

        __temp[pair] = {
            'Pair': pair,
            #'Summary': random.choice(all_summs)
            'Summary': __summ
        }

    summary_table_fakedata.append(__temp)
