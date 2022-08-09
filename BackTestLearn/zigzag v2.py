import pandas as pd 
import numpy as np 


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
def backtest(X, rise, output=True):
    # pivots = peak_valley_pivots(X, rise, -rise)
    allowance = 10000
    cash = 0
    shares = 0
    low = None
    high = None
    n = len(X)
    cash_array = np.zeros(n)
    shares_array = np.zeros(n)
    actions = np.zeros(n)
    up_thresh, down_thresh = rise, -rise
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
                
        # if last_pivot_t == t_n-1:
        #     pivots[last_pivot_t] = trend # 新趋势
        # elif pivots[t_n-1] == 0:
        #     pivots[t_n-1] = -trend # 老趋势
            
        p = X[i] # 当前价格
        
        # if pivots[i] == -1:
        #     low = p
            
        # if pivots[i] == 1:
        #     high = p
            
        if low is None or high is None: # 确保已经有了low high
            continue

        if p < low and shares >= 0: # and p < high * (1-rise) # TODO: 增加频率，避免一下子跌太多，错过做空时机
            # 做空
            shares = -allowance / p
            cash += allowance
            if output: print("做空", p, X.index[i])
            actions[i] = -1
            
        if (p > high) and shares < 0: # and p > low * (1 + rise)
        # if (p > (0.5 * low + 0.5 * high)) and shares < 0: # 特殊止损条件
            # 平仓止损
            cash += shares * p
            if output: print("止损", p, X.index[i], "盈亏", allowance + shares * p)
            shares = 0
            actions[i] = -2
        
        # if (p < 0.9 * low) and shares < 0:
        #     # 止盈
        #     cash += shares * p
        #     shares = 0
        #     if output: print("止盈", p, X.index[i])
            
        if i == len(pivots) - 1: # 最后时刻清仓
            cash += shares * p
            shares = 0
        
        cash_array[i] = cash
        shares_array[i] = shares
    
    ## 计算收益
    df_pnl = pd.DataFrame({"cash": cash_array, "shares": shares_array, "price": X, "pivots": pivots, "actions": actions})
    df_pnl.index = X.index
    df_pnl["pnl"] = df_pnl["cash"] + df_pnl["shares"] * df_pnl["price"]
    
    # 计算收益率等参数
    df_pnl["ret"] = df_pnl["pnl"].diff().fillna(0) / allowance
    rets = df_pnl.loc[df_pnl["ret"] != 0, "ret"]
    freq = (rets.index[1] - rets.index[0]).total_seconds() // 60
    time_length = rets.shape[0] * ((rets.index[1] - rets.index[0]).total_seconds() / 3600 / 24 / 365)
    total_return = df_pnl["pnl"][-1] / allowance
    yearly_return = total_return / time_length
    yearly_vol = df_pnl["ret"].std() * np.sqrt(60 / freq * 24 * 365)
    maxdrawdown = max(rets.cumsum().cummax() - rets.cumsum())
    sharpe = yearly_return / yearly_vol
    stats = dict(rise=rise, freq=freq, total_return=total_return, yearly_vol=yearly_vol, maxdrawdown=maxdrawdown, sharpe=sharpe, hold_time=time_length, start_date=X.index[0].date(), end_date=X.index[-1].date())
    # stats = [rise, total_return, yearly_vol, maxdrawdown]
    return df_pnl, pivots, actions, stats

# freq = 25
# rise = 0.05
start_date = "2022"
# data = df.resample(f'{freq}T').last()
# X = data["Close"].loc[start_date:]

stats_list = []  
def run_backtest(args):    
    rise, freq = args 
    # start_date="2020"
    print(rise, freq, start_date)
    X = df["Close"].resample(f'{freq}T').last().loc[start_date:]
    df_pnl, pivots, actions, stats = backtest(X, rise, False)
    # display(pd.DataFrame([stats]))
    return stats

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
def safe_arange(start, stop, step):
    return step * np.arange(start / step, stop / step)

def main():
    args = [(a, b) for a in safe_arange(0.001, 0.1, 0.001) for b in [1,2,3,5,10,15,20,25,30,60,120,240,480,960,1440]]
    
    start = time.perf_counter()
    with ProcessPoolExecutor(multiprocessing.cpu_count() - 1) as executor:
    # with ProcessPoolExecutor(10) as executor:
        stats_list = executor.map(run_backtest, args)
    end = time.perf_counter()
    print("Finished {} iterations in {} seconds".format(len(args), end - start))
    
    df_stats = pd.DataFrame(stats_list)
    df_stats = df_stats.sort_values("total_return", ascending=False)
    print(df_stats)
    df_stats.to_csv(f"zigzag_rank{start_date}v1.csv", index=False)
    
    
if __name__ == '__main__':
    main()