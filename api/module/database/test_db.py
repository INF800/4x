import database.models.forex as forex_model
from sqlalchemy.orm import Session
from database.engine_base_sess import SessionLocal, engine


def get_session(sess=SessionLocal()):
	""" returns db session """
	try: return sess
	finally:
		sess.close()


#------------------------------------------------------------------------------------------------------------
# create empty tables (and db)
#------------------------------------------------------------------------------------------------------------
forex_model.Base.metadata.create_all(bind=engine) 

"""
#------------------------------------------------------------------------------------------------------------
# add data to db
#------------------------------------------------------------------------------------------------------------
def add_currency_pair(data, sess=get_session()):
    '''
    data is a dictionary with:
        + pairName
        + price
        + change
        + perChange
    '''
    curPair = forex_model.CurPairs()
    curPair.cur_pair = data['pairName']
    curPair.price = data['price']
    curPair.change = data['change']
    curPair.per_change = data['perChange']
    
    sess.add(curPair)
    sess.commit()


add_currency_pair({
    'pairName'  : 'EUR/USD',
    'price'     : '10.100',
    'change'    : '+1.01',
    'perChange' : '+0.001',
})

'''
$ sqlite3 database.db

sqlite> .schema
sqlite> SELECT * FROM CurPairs;

1|2021-01-19 17:55:23.693424|EUR/USD|10.100|+1.01|+0.001
2|2021-01-19 17:59:30.928868|EUR/USD|10.100|+1.01|+0.001
'''
"""