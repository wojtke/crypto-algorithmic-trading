import pandas as pd
import os
from datetime import datetime
import shutil

trades = pd.DataFrame(columns=["Side", "Entry price", "Exit price", "Entry time", "Exit time", "Amount", "PNLrealised", "PNLmax", "PNLmin"])

class Trade:
    def __init__(self):
        self.clear()

    def open(self, side):
        self.side = side

    def add(self, entry_price, entry_time, amount, liq):
        self.entry_price.append(entry_price)
        self.entry_time.append(entry_time)
        self.amount.append(amount)
        self.liq = liq

    def print(self):
        print(  "Side ", self.side,
                "\nentry price ", self.entry_price,
                "\nentry time ", self.entry_time,
                "\namount ", self.amount,
                "\npnl_max ", self.pnl_max,
                "\npnl_min ", self.pnl_min)


    def update_pnl(self, pnl_high, pnl_low):
        if self.pnl_max<pnl_high:
           self.pnl_max=pnl_high
        if self.pnl_min>pnl_low:
           self.pnl_min=pnl_low

    def close(self, exit_price, exit_time, pnl_realised):
        global trades
        trades = trades.append([{"Side": self.side, 
                                "Entry price":self.entry_price, 
                                "Exit price":exit_price, 
                                "Entry time":self.entry_time, 
                                "Exit time":exit_time, 
                                "Amount": self.amount, 
                                "PNLrealised":pnl_realised, 
                                 "PNLmax": self.pnl_max, 
                                 "PNLmin": self.pnl_min}])
        self.clear()

    def liquidate(self, exit_time):
        global trades
        trades = trades.append([{"Side": self.side, 
                                "Entry price":self.entry_price, 
                                "Exit price":self.liq, 
                                "Entry time":self.entry_time, 
                                "Exit time":exit_time, 
                                "Amount": self.amount, 
                                "PNLrealised":-1, 
                                 "PNLmax": self.pnl_max, 
                                 "PNLmin": self.pnl_min}])
        self.clear()

    def clear(self):
        self.pnl_max = 0
        self.pnl_min = 0

        self.entry_price = []
        self.entry_time = []
        self.amount = []
        self.side = None


def save_trades(name, model, comment):
    global trades
    os.makedirs(os.path.join(os.path.dirname(__file__), f'STATISTICS\\{name}'))

    shutil.copy2('chart.py', f'STATISTICS\\{name}')

    trades.to_csv(os.path.join(os.path.dirname(__file__), f'STATISTICS\\{name}\\trades.csv'), index = False)


    f = open(f'STATISTICS\\{name}\\comment.txt','w')
    f.write(analyze()+"\n"+model+'\n'+comment)
    f.close()

    trades = pd.DataFrame(columns=["Side", "Entry price", "Exit price", "Entry time", "Exit time", "Amount", "PNLrealised", "PNLmax", "PNLmin"])


def fix_index():
    global trades
    trades = trades.reset_index()
    trades = trades.drop(columns=["index"])

def get_trades():
    fix_index()
    pd.set_option('display.max_columns', None)
    print(trades)

def analyze():
    liq=0
    loss=0
    profit=0
    break_even=0
    for trad in trades['PNLrealised']:
        if trad == -1:
            liq+=1
        elif trad < -0.001:
            loss+=1
        elif trad > 0.001:
            profit+=1
        else:
            break_even+=1

    if profit+loss+liq>0:
        stats=f"Profitable: {profit}, "
    if profit+loss+liq>0:
        stats+=f"({round(100*profit/(profit+loss+liq))} pct)"
    stats+=f'\nunprofitable: {loss+liq}, '
    if loss+liq>0:
        stats+= f'including {liq} liqs ({round(100*liq/(loss+liq))} pct of unprofitable is liq)\n'
    stats+=f"break-even: {break_even}\n"
    stats+=f"First entry: {datetime.fromtimestamp(int(trades['Entry time'].iloc[0][0]/1000))}"
    stats+=f"\nLast entry: {datetime.fromtimestamp(int(trades['Entry time'].iloc[-1][0]/1000))}"

    print(stats)
    return stats


