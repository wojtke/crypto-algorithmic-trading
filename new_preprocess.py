from data_processing import Preprocessor
from datetime import datetime

SYMBOL = 'BTC'
INTERVAL = '15m'
WINDOWS = None #if None, then default value from Vars

pasts = [100]
futures = [50]
target_pct = [0.25, 1]

special_name = 'ehh'

preprocessor = Preprocessor(SYMBOL, INTERVAL)

date_str = datetime.now().strftime("%d.%m.%y")
for p in pasts:
    for f in futures:
        for t_pct in target_pct:

            preprocessor.preprocess(TARGET_CHANGE=t_pct/100, 
	            					PAST_SEQ_LEN=p, 
	            					FUTURE_CHECK_LEN=f, 
	            					SPECIAL_NAME=special_name,
	            					date_str=date_str)





