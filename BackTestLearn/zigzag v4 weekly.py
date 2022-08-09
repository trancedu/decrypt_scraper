import pandas as pd 
import numpy as np 
import os


df = pd.read_csv("BackTestLearn/BTC_Close.csv", index_col=0)
df.index = pd.to_datetime(df.index, unit='ms')
data = df.resample('15T').last()


# def plot_pivots(X, pivots, ax=None):
#     if ax:
#         # ax.ylim(X.min()*0.99, X.max()*1.01)
#         ax.plot(X, 'k:', alpha=0.5)
#         ax.plot(X[pivots != 0], 'k-')
#         ax.scatter(X[pivots == 1].index, X[pivots == 1], color='g')
#         ax.scatter(X[pivots == -1].index, X[pivots == -1], color='r')
#         return
        
#     plt.figure(figsize=(20, 10))
#     # plt.xlim(0, len(X))
#     plt.ylim(X.min()*0.99, X.max()*1.01)
#     plt.plot(X, 'k:', alpha=0.5)
#     plt.plot(X[pivots != 0], 'k-')
#     plt.scatter(X[pivots == 1].index, X[pivots == 1], color='g')
#     plt.scatter(X[pivots == -1].index, X[pivots == -1], color='r')


"""
reference:
https://github.com/jbn/ZigZag.git
"""
import numpy as np

PEAK = 1
VALLEY = -1


def identify_initial_pivot(X, up_thresh, down_thresh):
    x_0 = X[0]
    x_t = x_0

    max_x = x_0
    min_x = x_0

    max_t = 0
    min_t = 0

    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X)-1
    return VALLEY if x_0 < X[t_n] else PEAK


def peak_valley_pivots(X, up_thresh, down_thresh):
    """
    Find the peaks and valleys of a series.

    :param X: the series to analyze
    :param up_thresh: minimum relative change necessary to define a peak
    :param down_thesh: minimum relative change necessary to define a valley
    :return: an array with 0 indicating no pivot and -1 and 1 indicating
        valley and peak


    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias
    analysis.
    """
    if down_thresh > 0:
        raise ValueError('The down_thresh must be negative.')

    initial_pivot = identify_initial_pivot(X, up_thresh, down_thresh)
    t_n = len(X)
    pivots = np.zeros(t_n, dtype=np.int_)
    trend = -initial_pivot
    last_pivot_t = 0
    last_pivot_x = X[0]

    pivots[0] = initial_pivot

    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.
    up_thresh += 1
    down_thresh += 1

    for t in range(1, t_n):
        x = X[t]
        r = x / last_pivot_x

        if trend == -1:
            if r >= up_thresh:
                pivots[last_pivot_t] = trend # 上一个被标记为谷底的点
                trend = PEAK
                last_pivot_x = x
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            if r <= down_thresh:
                pivots[last_pivot_t] = trend
                trend = VALLEY
                last_pivot_x = x
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t

    if last_pivot_t == t_n-1:
        pivots[last_pivot_t] = trend # 新趋势
    elif pivots[t_n-1] == 0:
        pivots[t_n-1] = -trend # 老趋势

    return pivots


def max_drawdown(X):
    """
    Compute the maximum drawdown of some sequence.

    :return: 0 if the sequence is strictly increasing.
        otherwise the abs value of the maximum drawdown
        of sequence X
    """
    mdd = 0
    peak = X[0]

    for x in X:
        if x > peak:
            peak = x

        dd = (peak - x) / peak

        if dd > mdd:
            mdd = dd

    return mdd if mdd != 0.0 else 0.0


def pivots_to_modes(pivots):
    """
    Translate pivots into trend modes.

    :param pivots: the result of calling ``peak_valley_pivots``
    :return: numpy array of trend modes. That is, between (VALLEY, PEAK] it
    is 1 and between (PEAK, VALLEY] it is -1.
    """

    modes = np.zeros(len(pivots), dtype=np.int_)
    mode = -pivots[0]

    modes[0] = pivots[0]

    for t in range(1, len(pivots)):
        x = pivots[t]
        if x != 0:
            modes[t] = mode
            mode = -x
        else:
            modes[t] = mode

    return modes


def compute_segment_returns(X, pivots):
    """
    :return: numpy array of the pivot-to-pivot returns for each segment."""
    pivot_points = np.array(X[pivots != 0])
    return pivot_points[1:] / pivot_points[:-1] - 1.0


# TODO: 加上均线
# 可能需要两套体系来存放
# 买入卖出都设置为1BTC
def backtest_both(X, rise, down, take_profit_long=10, take_profit_short=0, output=True):
    # pivots = peak_valley_pivots(X, rise, -rise)
    allowance = 1
    cash = 0
    shares_long = 0
    shares_short = 0
    low = None
    high = None
    n = len(X)
    cash_array = np.zeros(n)
    shares_long_array = np.zeros(n)
    shares_short_array = np.zeros(n)
    actions_long = np.zeros(n)
    actions_short = np.zeros(n)
    cum_returns = np.zeros(n)
    up_thresh, down_thresh = rise, down
    last_buy = 1000000
    last_sell = 1000000
    last_price = 0
    for i in range(n):
        if i == 0:
            initial_pivot = identify_initial_pivot(X[:100], up_thresh, down_thresh) # 这个会数据泄漏
            t_n = len(X)
            pivots = np.zeros(t_n, dtype=np.int_)
            trend = -initial_pivot
            last_pivot_t = 0
            last_pivot_x = X[0]

            pivots[0] = initial_pivot
            up_thresh += 1
            down_thresh += 1
            last_price = X[i]
            continue
        
        t = i
        x = X[t]
        r = x / last_pivot_x

        if trend == -1:
            if r >= up_thresh: # 如果上升幅度超过阈值，则标记上个点为谷底
                pivots[last_pivot_t] = trend # 上一个被标记为谷底的点
                low = last_pivot_x
                trend = PEAK
                last_pivot_x = x
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            if r <= down_thresh: # 如果下降幅度超过阈值，则标记上个点为顶点
                pivots[last_pivot_t] = trend
                high = last_pivot_x
                trend = VALLEY
                last_pivot_x = x
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t   
            
        p = X[i] # 当前价格
            
        if low is None or high is None: # 确保已经有了low high
            continue
        
        ## 计算收益率
        cur_return = 0 
        if shares_long > 0:        
            cur_return += (p - last_price) / last_buy
        if shares_short < 0:
            cur_return += (last_price - p) / last_sell
        cum_returns[i] = cur_return + cum_returns[i - 1]
        last_price = p
        
        if (p < low) and shares_long > 0: # 如果当前价格小于谷底，并且有持仓，则卖出止损
            # 平仓止损
            cash += shares_long * p
            if output: print("止损", p, X.index[i], "盈亏", p - last_buy)
            shares_long = 0
            actions_long[i] = 2
            last_sell = p
        
        # if (p > take_profit_long * last_buy) and shares > 0: # 如果当前价格大于买入价格的止盈，并且有持仓，则卖出止盈
        #     # 止盈
        #     cash += shares * p
        #     if output: print("止盈", p, X.index[i], "盈亏", allowance + shares * p)
        #     shares = 0
        #     actions[i] = 3
        #     last_sell = p
        
        if (p > high) and shares_short < 0: # 如果当前价格大于顶点，并且有做空持仓，则买入止损
            # 平仓止损
            cash += shares_short * p
            if output: print("止损", p, X.index[i], "盈亏", last_sell - p)
            shares_short = 0
            actions_short[i] = -2
            last_buy = p


            
        
        # if (p < take_profit_short * last_sell) and shares < 0: # 如果当前价格小于卖出价格的止盈，并且有做空，则买入止盈
        #     # 止盈
        #     cash += shares * p
        #     if output: print("止盈", p, X.index[i], "盈亏", allowance + shares * p)
        #     shares = 0
        #     actions[i] = -3
        #     last_buy = p
            
        if p > high and shares_long == 0: # 如果当前价格大于顶点，并且没有持仓，则做多
            # 做多
            shares_long = allowance
            cash -= shares_long * p
            if output: print("做多", p, X.index[i])
            actions_long[i] = 1
            last_buy = p
        
        if p < low and shares_short == 0: # 如果当前价格小于谷底，并且没有持仓，则做空
            # 做空
            shares_short = -allowance
            cash -= shares_short * p
            if output: print("做空", p, X.index[i])
            actions_short[i] = -1
            last_sell = p
            
            
        if i == len(pivots) - 1: # 最后时刻清仓
            cash += (shares_short + shares_long) * p
            shares_short = shares_long = 0
        
        cash_array[i] = cash
        shares_long_array[i] = shares_long
        shares_short_array[i] = shares_short
    
    shares_array = shares_long_array + shares_short_array
    if last_pivot_t == t_n-1:
        pivots[last_pivot_t] = trend # 新趋势
    elif pivots[t_n-1] == 0:
        pivots[t_n-1] = -trend # 老趋势
    
    ## 计算收益
    df_pnl = pd.DataFrame({"cash": cash_array, "shares": shares_array, "actions_long": actions_long, "actions_short": actions_short, "price": X, "pivots": pivots})
    df_pnl.index = X.index
    df_pnl["pnl"] = df_pnl["cash"] + df_pnl["shares"] * df_pnl["price"]
    df_pnl["cum_return"] = cum_returns
    
    # 计算收益率等参数
    df_pnl["ret"] = df_pnl["cum_return"].diff().fillna(0)
    freq = (df_pnl.index[1] - df_pnl.index[0]).total_seconds() // 60
    rets = df_pnl.loc[df_pnl["ret"] != 0, "ret"]
    if len(rets) == 0:
        stats = {}
    else:
        time_length = rets.shape[0] * ((df_pnl.index[1] - df_pnl.index[0]).total_seconds() / 3600 / 24 / 365)
        total_return = df_pnl["cum_return"][-1]
        yearly_return = total_return / time_length
        yearly_vol = df_pnl["ret"].std() * np.sqrt(60 / freq * 24 * 365)
        maxdrawdown = max(rets.cumsum().cummax() - rets.cumsum())
        sharpe = yearly_return / yearly_vol
        stats = dict(rise=rise, down=down, freq=freq, take_profit_long=take_profit_long, take_profit_short=take_profit_short, total_return=total_return, yearly_vol=yearly_vol, maxdrawdown=maxdrawdown, sharpe=sharpe, hold_time=time_length, start_date=X.index[0].date(), end_date=X.index[-1].date())
    # stats = [rise, total_return, yearly_vol, maxdrawdown]
    return df_pnl, stats


import datetime

start = "2020-01-01"
end = "2022-07-01"
start_date = start


# freq = 25
# rise = 0.05
# start_date = "2019"
# data = df.resample(f'{freq}T').last()
# X = data["Close"].loc[start_date:]

stats_list = []  
def run_backtest(args):    
    rise, down, freq, take_profit = args 
    print(rise, down, freq, take_profit, start, end)
    data = df.resample(f'{freq}T').last().fillna(method="ffill")
    stats_list = []
    for start_date in pd.date_range(start=start, end=end, freq="W-FRI"):
        end_date = start_date + datetime.timedelta(days=7)

        X = data["Close"].loc[start_date:end_date]
        df_pnl, stats = backtest_both(X, rise, down, take_profit, output=False)
        stats_list.append(stats)

    df_stats = pd.DataFrame(stats_list)
    mean_stats = df_stats.select_dtypes(include=["float","int"]).mean()
    mean_stats["total_return"] = df_stats["total_return"].sum()
    mean_stats["strategy_pct"] = 1 - df_stats.isna().sum().max() / df_stats.shape[0]
    
    return mean_stats

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
def safe_arange(start, stop, step):
    return step * np.arange(start / step, stop / step)

def main():
    # args = [(a, c, b) for a in safe_arange(0.01, 0.12, 0.01) for c in safe_arange(-0.11, 0, 0.01) for b in [1,2,3,5,10,15,20,25,30,60,120,240,480,960,1440]]
    # args = [(a, c, b) for a in safe_arange(0.01, 0.12, 0.01) for c in safe_arange(-0.11, 0, 0.01) for b in 1440 * safe_arange(1, 11, 1)]
    # args = [(a, c, b, d) for a in safe_arange(0.01, 0.12, 0.01) for c in safe_arange(-0.11, 0, 0.01) for b in [720, 1440, 2160,7200] for d in [0]+safe_arange(0.6, 1, 0.05).tolist()]
    # args = [(a, c, b, d) for a in safe_arange(0.01, 0.12, 0.01) for c in safe_arange(-0.11, 0, 0.01) for b in [720, 1440, 2160,7200] for d in [10]+safe_arange(1, 1.6, 0.1).tolist()]
    args = [(a, c, b, d) for a in safe_arange(0.01, 0.08, 0.01) for c in safe_arange(-0.08, -0.002, 0.01) for b in [5, 15, 30, 60] for d in [1]]
    # args = [(a, c, b, d) for a in safe_arange(0.01, 0.12, 0.01) for c in safe_arange(-0.11, 0, 0.01) for b in [15] for d in [1]]
    
    start = time.perf_counter()
    with ProcessPoolExecutor(multiprocessing.cpu_count() - 1) as executor:
    # with ProcessPoolExecutor(10) as executor:
        stats_list = executor.map(run_backtest, args)
    end = time.perf_counter()
    print("Finished {} iterations in {} seconds".format(len(args), end - start))
    
    df_stats = pd.DataFrame(stats_list).dropna()
    df_stats = df_stats.sort_values("total_return", ascending=False)
    print(df_stats)
    
    ## 保存结果
    i = 1
    while True:
        file_name = f"BackTestLearn/zigzag_rank{start_date}_updown_profit_both_weekly v{i}.csv"
        if os.path.exists(file_name):
            i += 1 
        else:
            df_stats.to_csv(file_name, index=False)
            break
   
    
if __name__ == '__main__':
    main()