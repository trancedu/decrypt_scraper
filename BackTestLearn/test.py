def backtest_both(X, rise, down, take_profit_long=10, take_profit_short=0, output=True, mode=0, ma_array=None, ma_window=0):
    """_summary_

    Args:
        X (_type_): _description_
        rise (_type_): _description_
        down (_type_): _description_
        take_profit_long (int, optional): _description_. Defaults to 10.
        take_profit_short (int, optional): _description_. Defaults to 0.
        output (bool, optional): _description_. Defaults to True.
        mode (int, optional): 0 for both, 1 for long, 2 for short. Defaults to 0. 
        ma_array (_type_, optional): _description_. Defaults to None.
        ma_window (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    allowance = 1 # 每次交易允许买1个比特币，请勿修改，若修改得改变后面收益率计算方式
    cash = 0 # 初始化资金
    shares_long = 0 # 初始化多头持仓
    shares_short = 0 # 初始化空头持仓
    low = high = None # 初始化最高价和最低价
    n = len(X) # 数据长度
    cash_array = np.zeros(n) # 初始化资金数组
    shares_long_array = np.zeros(n) # 初始化多头持仓数组
    shares_short_array = np.zeros(n) # 初始化空头持仓数组
    actions_long = np.zeros(n) # 初始化多头仓位变化数组
    actions_short = np.zeros(n) # 初始化空头仓位变化数组
    cum_returns = np.zeros(n) # 初始化累计收益数组
    up_thresh, down_thresh = rise, down # 设置上涨下跌阈值
    last_buy = 1000000 # 初始化上一次买入价格
    last_sell = 1000000 # 初始化上一次卖出价格
    last_price = 0 # 初始化上一次价格
    
    # 回测
    for t in range(n):
        # 第一天找到初始趋势，高点和低点
        if t == 0:
            initial_pivot = identify_initial_pivot(X[:100], up_thresh, down_thresh) # 这个会数据泄漏
            t_n = len(X)
            pivots = np.zeros(t_n, dtype=np.int_)
            trend = -initial_pivot
            last_pivot_t = 0
            last_pivot_x = X[0]

            pivots[0] = initial_pivot
            up_thresh += 1
            down_thresh += 1
            last_price = X[t]
            continue
        
        
        x = X[t] # 当前价格
        r = x / last_pivot_x # 当前价格与上一个最高点/最低点的比例

        # 判断趋势，高点，低点
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
            
        p = X[t] # 当前价格
        ma_level = ma_array[t] if ma_array and not np.isnan(ma_array[t]) else 0 # 当前均线
            
        if low is None or high is None: # 确保已经有了low high再开始交易
            continue
        
        ## 计算收益率
        cur_return = 0 
        if shares_long > 0:        
            cur_return += (p - last_price) / last_buy # 多头今天变化的收益率
        if shares_short < 0:
            cur_return += (last_price - p) / last_sell # 空头今天变化的收益率
        # 当前累计收益率 = 累计到昨天收益率 + 今天变化的收益率
        cum_returns[t] = cur_return + cum_returns[t - 1]
        last_price = p # 更新上一次价格
        
        if mode >= 0 and (p < low) and shares_long > 0: # 如果当前价格小于谷底，并且有持仓，则卖出止损
            # 平仓止损
            # cash += shares_long * p
            cash += shares_long * low
            if output: print("止损", p, X.index[t], "盈亏", p - last_buy)
            shares_long = 0
            actions_long[t] = 2 # 卖出止损
            # last_sell = p
        
        # if (p > take_profit_long * last_buy) and shares > 0: # 如果当前价格大于买入价格的止盈，并且有持仓，则卖出止盈
        #     # 止盈
        #     cash += shares * p
        #     if output: print("止盈", p, X.index[i], "盈亏", allowance + shares * p)
        #     shares = 0
        #     actions[i] = 3
        #     last_sell = p
        
        if mode <= 0 and (p > high) and shares_short < 0: # 如果当前价格大于顶点，并且有做空持仓，则买入止损
            # 平仓止损
            # cash += shares_short * p
            cash += shares_short * high
            if output: print("止损", p, X.index[t], "盈亏", last_sell - p)
            shares_short = 0
            actions_short[t] = -2 # 买入止损
            # last_buy = p


            
        
        # if (p < take_profit_short * last_sell) and shares < 0: # 如果当前价格小于卖出价格的止盈，并且有做空，则买入止盈
        #     # 止盈
        #     cash += shares * p
        #     if output: print("止盈", p, X.index[i], "盈亏", allowance + shares * p)
        #     shares = 0
        #     actions[i] = -3
        #     last_buy = p
        
        if mode >= 0 and p > high and shares_long == 0 and (ma_level == 0 or p > ma_level): # 如果当前价格大于顶点，并且没有持仓，则做多
            # 做多
            shares_long = allowance
            # cash -= shares_long * p
            cash -= shares_long * high
            if output: print("做多", p, X.index[t])
            actions_long[t] = 1
            last_buy = p
        
        if mode <= 0 and p < low and shares_short == 0 and (ma_level == 0 or p < ma_level): # 如果当前价格小于谷底，并且没有持仓，则做空
            # 做空
            shares_short = -allowance
            # cash -= shares_short * p
            cash -= shares_short * low
            if output: print("做空", p, X.index[t])
            actions_short[t] = -1
            last_sell = p
            
            
        if t == len(pivots) - 1: # 最后时刻清仓
            cash += (shares_short + shares_long) * p
            shares_short = shares_long = 0
        
        cash_array[t] = cash
        shares_long_array[t] = shares_long
        shares_short_array[t] = shares_short
    
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
    rets = df_pnl.loc[df_pnl["ret"] != 0, "ret"] # 只计算入场的时刻的收益率
    if len(rets) == 0:
        stats = {}
    else:
        time_length = rets.shape[0] * ((df_pnl.index[1] - df_pnl.index[0]).total_seconds() / 3600 / 24 / 365)
        total_return = df_pnl["cum_return"][-1]
        yearly_return = total_return / time_length
        yearly_vol = df_pnl["ret"].std() * np.sqrt(60 / freq * 24 * 365)
        maxdrawdown = max(rets.cumsum().cummax() - rets.cumsum())
        sharpe = yearly_return / yearly_vol
        stats = dict(rise=rise, down=down, freq=freq, take_profit_long=take_profit_long, take_profit_short=take_profit_short, total_return=total_return, yearly_vol=yearly_vol, maxdrawdown=maxdrawdown, sharpe=sharpe, hold_time=time_length, ma_window=ma_window, start_date=X.index[0].date(), end_date=X.index[-1].date())
    # stats = [rise, total_return, yearly_vol, maxdrawdown]
    return df_pnl, stats