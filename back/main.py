# --------------------------------------------------------------------------------------------------
# create fastapi app 
# --------------------------------------------------------------------------------------------------
from fastapi import FastAPI
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)
# --------------------------------------------------------------------------------------------------
# end: create fastapi app 
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# define structure for requests (Pydantic & more)
# --------------------------------------------------------------------------------------------------
from fastapi import Request # for get
from pydantic import BaseModel # for post
	
class LiveBarRequest(BaseModel):
    msg: str
# --------------------------------------------------------------------------------------------------
# end: define structure for requests (Pydantic & more)
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# investing.com summary scraper 
# --------------------------------------------------------------------------------------------------
from scraper.investing.summary_table_scraper import uri as investing_uri
from scraper.investing.summary_table_scraper import SummaryTableScraper, proc_pair_info, PairScores

scraper = SummaryTableScraper(uri=investing_uri, class_name='technicalSummaryTbl')
pair_scores = PairScores()
# --------------------------------------------------------------------------------------------------
# end: investing.com summary scraper 
# --------------------------------------------------------------------------------------------------


# ==================================================================================================
# routes
# ==================================================================================================
@app.post("/livebardata")
def get_live_bar_data(req: LiveBarRequest):
    """
    returns reatime data for d3 chart
    """
    scraper.goto(investing_uri)
    data = scraper.get_pairs_info()
    chart_data = []

    for _, pair_info in data.items():
        strong = proc_pair_info(pair_info)
        
        strong_buy_or_sell = "" 
        if strong: 
            pair_scores.increment(pair_info['Pair'])
            strong_buy_or_sell = pair_info['Summary'][0]
        elif not strong: 
            pair_scores.decrement(pair_info['Pair'])

        chart_data.append({
            'key': pair_info['Pair'] + " " + strong_buy_or_sell,
            'value': pair_scores.scores[pair_info['Pair']]
        })

    return chart_data