from binance.enums import *
from binance.client import Client
import pandas as pd

class Zagranie:
    def __init__(self, orders, side, start_order_id, oco_limit_price, oco_stop_price, PARTIAL_FILL_TIME=60):
        self.orders = orders
        self.side = side 
        self.state = "new"
        self.PARTIAL_FILL_TIME = PARTIAL_FILL_TIME

        self.start_order_id = start_order_id

        self.oco_limit_id = None
        self.oco_limit_price = oco_limit_price
        self.oco_stop_id = None
        self.oco_stop_price = oco_stop_price
        self.time = None


        def update(self):
            if self.state == 'new':
                order_details = self.orders.client.futures_get_order(symbol=self.orders.SYMBOL, orderId=self.start_order_id)
                if order_details['status'] == "NEW":
                    pass
                elif order_details['status'] == "FILLED":
                    self.open(order_details['executedQty'])
                elif order_details['status'] == "PARTIALLY_FILLED":  
                    self.state = 'new_pf'
                    self.time = time.time()
                else:
                    self.close()
                    print(f"START LIMIT ORDER STATUS: {order_details['status']}")

            elif self.state == 'open':
                order_details_oco_limit = self.orders.client.futures_get_order(symbol=self.orders.SYMBOL, orderId=self.oco_limit_id)
                order_details_oco_stop = self.orders.client.futures_get_order(symbol=self.orders.SYMBOL, orderId=self.oco_stop_id)

                if order_details_oco_limit['status'] == "NEW" and order_details_oco_stop['status'] == "NEW":
                    pass
                if order_details_oco_limit['status'] == "FILLED" or order_details_oco_stop['status'] == "FILLED":
                    self.state = "closed"
                elif order_details_oco_limit['status'] == "PARTIALLY_FILLED":
                    self.state = 'closing_pf'
                    self.time = time.time()
                else:
                    self.close()
                    print(f"OCO LIMIT STATUS: {order_details_oco_limit['status']}")
                    print(f"OCO STOP STATUS: {order_details_oco_stop['status']}")

            elif self.state == 'new_pf':
                order_details = self.orders.client.futures_get_order(symbol=self.orders.SYMBOL, orderId=self.start_order_id)
                if order_details['status'] == "PARTIALLY_FILLED":
                    if self.time + self.PARTIAL_FILL_TIME > time.time():
                        self.open(order_details['executedQty'])
                        self.market_order(order_details['origQty'] - order_details['executedQty'], self.orders.side_reversed[self.side])
                elif order_details['status'] == "FILLED":
                    self.open(order_details['executedQty'])
                else:
                    self.close()
                    print(f"START LIMIT ORDER STATUS: {order_details['status']}")
                    
            elif self.state == 'closing_pf':
                order_details = self.orders.client.futures_get_order(symbol=self.orders.SYMBOL, orderId=self.oco_limit_id)
                if order_details['status'] == "PARTIALLY_FILLED":
                    if self.time + self.PARTIAL_FILL_TIME > time.time():
                        self.close(order_details['origQty'] - order_details['executedQty'])
                elif order_details['status'] == "FILLED":
                    self.state = "closed"
                else:
                    self.close()
                    print(f"OCO LIMIT STATUS: {order_details_oco_limit['status']}")

            print(f"State: {self.state}")


        def open(self, quantity):
            self.state = 'open'

            order_limit = self.orders.client.futures_create_order(
                                symbol = self.orders.SYMBOL+'USDT',
                                side = self.orders.side_reversed[self.side],
                                type = 'LIMIT',
                                timeInForce = TIME_IN_FORCE_GTC,
                                quantity = quantity,
                                price = self.oco_limit_price,
                                newOrderRespType = "RESULT") 

            order_stop = self.orders.client.futures_create_order(
                                symbol = self.orders.SYMBOL+'USDT',
                                side = self.orders.side_reversed[self.side],
                                type = 'STOP_MARKET',
                                quantity = quantity,
                                stopPrice = self.oco_stop_price,
                                newOrderRespType = "RESULT") 

            self.oco_limit_id = order_limit['orderId']
            self.oco_stop_id = order_stop['orderId']


        def market_order(self, quantity, side):
            market_order = self.orders.client.futures_create_order(
                                    symbol = self.order.SYMBOL+'USDT',
                                    side = side,
                                    type = 'MARKET',
                                    quantity = float(quantity), #JAKIS DOMYSLMY PRICE TRZRBA DAC
                                    newOrderRespType = "RESULT")


        def close(self, quantity=self.orders.positionInfo()['positionAmt']):
            self.update()
            if self.state == 'new':
                self.orders.client.futures_cancel_order(symbol=self.orders.SYMBOL, orderId=self.start_order_id)
            elif self.state == 'open':
                self.orders.client.futures_cancel_order(symbol=self.orders.SYMBOL, orderId=self.oco_limit_id)
                self.orders.client.futures_cancel_order(symbol=self.orders.SYMBOL, orderId=self.oco_stop_id)

                self.market_order(quantity, self.orders.side_reversed[self.side])

            elif self.state == 'new_pf':
                self.orders.client.futures_cancel_order(symbol=self.orders.SYMBOL, orderId=self.start_order_id)

                self.market_order(quantity, self.orders.side_reversed[self.side])

            elif self.state == 'closing_pf':
                self.orders.client.futures_cancel_order(symbol=self.orders.SYMBOL, orderId=self.oco_limit_id)
                self.orders.client.futures_cancel_order(symbol=self.orders.SYMBOL, orderId=self.oco_stop_id)

                self.market_order(quantity, self.orders.side_reversed[self.side])  

            elif self.state == "closed":
                print("Closed already")

            self.state = 'closed'


class Orders:
    def __init__(self, SYMBOL, LEWAR, PYRAMID_MAX, ORDER_SIZE, is_it_for_real=False):
        self.SYMBOL = SYMBOL
        self.LEWAR  = LEWAR
        self.PYRAMID_MAX = PYRAMID_MAX 
        self.ORDER_SIZE = ORDER_SIZE

        self.side = {"LONG":'BUY', "SHORT":'SELL'}
        self.side_reversed = {"SHORT": 'BUY', "LONG":'SELL', "BUY":'SELL', "SELL":'BUY'}

        if is_it_for_real:
            with open("api_keys.txt") as d: 
                api_keys = d.readlines()
                api_keys[0]=api_keys[0].strip()
            self.client = Client(api_keys[0], api_keys[1])


        self.zagranie_list = []

        
        try:
            client.futures_change_margin_type(symbol=self.SYMBOL+'USDT', marginType="ISOLATED")
        except: 
            pass

        self.client.futures_change_leverage(symbol=self.SYMBOL+'USDT', leverage=self.LEWAR)
        
    def update(self):
        l = len(self.zagranie_list)
        for i, zagranie in enumerate(self.zagranie_list[::-1]):
            zagranie.update()
            if zagranie.state == 'closed':
                zagranie.close()
                del zagranie_list[l-1-i]
                

    def order_size(self, price, balance):
        #return 0.6**pyramid*ORDER_SIZE*balance*LEWAR/price
        return self.ORDER_SIZE*balance*self.LEWAR/price

    def get_balance(self):  
        a = self.client.futures_account_balance()
        for e in a:
            if e['asset']=='USDT':
                return float(e['withdrawAvailable'])

    def positionInfo(self):
        a = self.client.futures_position_information()
        for e in a:
            if e['symbol']==self.SYMBOL+'USDT':
                return e

    def create_order(self, side, price=None, TARGET_CHANGE=True):
        print(f"Opening {side}")

        if len(self.zagranie_list) >= self.PYRAMID_MAX:
            print("Pyramid max reached")
            return None

        balance = self.get_balance()

        if price:
            order = self.client.futures_create_order(
                            symbol = self.SYMBOL+'USDT',
                            side = self.side[side],
                            type = 'LIMIT',
                            timeInForce = TIME_IN_FORCE_GTC,
                            quantity = self.order_size(price, balance),
                            price = price,
                            newOrderRespType = "RESULT")    
        else:
            order = self.client.futures_create_order(
                            symbol = SYMBOL+'USDT',
                            side = self.side[side],
                            type = 'MARKET',
                            quantity = self.order_size(self.price, balance), #JAKIS DOMYSLMY PRICE TRZRBA DAC
                            newOrderRespType = "RESULT")


        if TARGET_CHANGE:
            if side == "LONG":
                oco_limit_price = self.price*(1+TARGET_CHANGE)
                oco_stop_price = self.price*(1-TARGET_CHANGE)
            elif side =="SHORT":
                oco_limit_price = self.price*(1-TARGET_CHANGE)
                oco_stop_price = self.price*(1+TARGET_CHANGE)

        zagranie = Zagranie(self, side, order['orderId'], oco_limit_price, oco_stop_price)

        self.zagranie_list.append(zagranie)

        return order['orderId']

    def close(self, side):
        print(f"Closing {side}")

        l = len(self.zagranie_list)
        for i, zagranie in enumerate(self.zagranie_list[::-1]):
            if zagranie.side == side:
                zagranie.close()
                del zagranie_list[l-1-i]


                
