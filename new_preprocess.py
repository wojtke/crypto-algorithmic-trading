from data_processing import Preprocessor
from datetime import datetime

SYMBOL = 'BTC'
INTERVAL = '15m'
WINDOWS = None #default

pasts = [200]
futures = [100]
target_pct = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 2]
target_pct = [1]

special_name = 'new'

preprocessor = Preprocessor(SYMBOL, INTERVAL)

date_str = datetime.now().strftime("%d.%m.%y")
for p in pasts:
    for f in futures:
        for t_pct in target_pct:

            #log = Logger([SYMBOL, INTERVAL, SPECIAL_NAME, PAST_SEQ_LEN, FUTURE_CHECK_LEN, TARGET_CHANGE])
            preprocessor.preprocess(TARGET_CHANGE=t_pct/100, 
	            					PAST_SEQ_LEN=p, 
	            					FUTURE_CHECK_LEN=f, 
	            					SPECIAL_NAME=special_name,
	            					date_str=date_str)


            #log.save(f'PREP_LOG', f'{INTERVAL}-{SPECIAL_NAME}-{date_str}.csv')



