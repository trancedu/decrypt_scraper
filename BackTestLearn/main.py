from tracemalloc import start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts
ts.set_token('e32fe83c30637d56260e179d663c8881c1a9af62a1bf5518a20627bb')
pro = ts.pro_api()

trade_cal = pro.trade_cal()

class Context:
    def __init__(self, cash, start_date, end_date):
        self.cash = cash
        self.start_date = start_date
        self.end_date = end_date
        self.positions = {} # 持仓
        self.benchmark = None # 基准
print(trade_cal)
        