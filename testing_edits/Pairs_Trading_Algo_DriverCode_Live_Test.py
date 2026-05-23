import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, kpss
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings('ignore')

# Step 1:

def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n)) 
    pvalue_matrix = np.ones((n, n)) 
    keys = data.keys() 
    pairs = [] 
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]] 
            S2 = data[keys[j]] 
            result = coint(S1, S2, autolag='AIC') 
            score = result[0] 
            pvalue = result[1] 
            score_matrix[i, j] = score 
            pvalue_matrix[i, j] = pvalue 
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j])) 
    return score_matrix, pvalue_matrix, pairs 

def calculate_hurst_exponent(time_series, max_lags=20):

	"""
	Calculates the Hurst Exponent (H) to evaluate statistical memory
	H < .05: Mean-reverting series
	H = .05: Ranom Walk
	H > .05: Trending Series
    	"""
	lags = range(2, max_lags)
	tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
	poly = np.polyfit(np.log(lags), np.log(tau), 1)
	return poly[0] * 2.0

def calculate_half_life(spread):
	"""
	Calculates the Half-Life (tau) of mean reversion using Ornstien-Uhlenbeck process.
	Returns the exact number of trading days it takes for a spread to revert halfway
	"""
	spread_lag = spread.shift(1)
	spread_diff = spread.diff()

	# Drop NaNs created by shifting
	valid_idx = spread_lag.notna() & spread_diff.notna()
	if not valid_idx.any():
		return np.nan

	X = sm.add_constant(spread_lag[valid_idx])
	y = spread_diff[valid_idx]

	model = sm.OLS(y, X).fit()
	lambda_param = model.params.iloc[1]

	# If lambda is positive, the series is diverging (not mean-reverting)
	if lambda_param >= 0:
		return np.inf

	half_life = -np.log(2) / lambda_param
	return half_life

def pair_filter(S1_log, S2_log, max_half_life=60):
	# Runs 5 Stage Testing gauntlet using short-circuit valuation for computational efficiency

	# Stage 1: Engle-Granger Filter
	eg_result = coint(S1_log, S2_log, autolag='AIC')
	eg_pvalue = eg_result[1]

	if eg_pvalue >= .05:
		return False, {} #Short-Circuit: Fail fast

	# Stage 2: Johansen Test (Confirmation)

	pair_df = pd.concat([S1_log, S2_log], axis=1)
	try:
		jres = coint_johansen(pair_df, det_order=0, k_ar_diff=1)
		trace_stat = jres.lr1[0]
		crit_value_95 = jres.cvt[0, 1] # 1 corresponds to 95% confidence boundry

		if trace_stat <= crit_value_95:
			return False, {} # Short-Circuit: Failed
	except Exception:
		return false, {}

	S2_with_const = sm.add_constant(S2_log)
	ols_model = sm.OLS(S1_log, S2_with_const).fit()
	beta = ols_model.params.iloc[1]
	spread = S1_log - (beta * S2_log)

	# Stage 3: Hurst Exponent (The memory Check)
	hurst_val = calculate_hurst_exponent(spread.values)
	if hurst_val >= 0.50:
		return False, {} # Short-Circuit: Spread is random or trending

	# Stage 4: KPSS Test (Confirmatory Stationarity)

	try:
		_, kpss_pvalue, _, _ = kpss(spread, regression='c', nlags="auto")
		if kpss_pvalue < 0.05:
			return False, {} # Short-Circuit: We reject stationarity
	except Exception:
		return False, {}

	# Stage 5: Half-Life Filter (Capital Velocity Check)

	half_life = calculate_half_life(spread)
	if np.isinf(half_life) or half_life <= 1 or half_life > max_half_life:
		return False, {} # Short-Circuit: Reversion takes too long; capital will be trapped

	# --- All Checks Passed ---
	metadata = {
		'Beta': beta,
		'EG_PValue': eg_pvalue,
		'Johansen_Trace': trace_stat,
		'Hurst': hurst_val,
		'KPSS_PValue': kpss_pvalue,
		'Half_Life_Days': half_life
	}
	return True, metadata


# Step 2:

def generate_execution_states(S1, S2, window=30, entry_thresh=2.0, exit_thresh=0.0):
	#S1: Dependent Asset
	#S2: Independent Asset

	# 1. Dynamic Spread Calculation (OLS Baseline)
	S2_with_const = sm.add_constant(S2)
	model = sm.OLS(S1, S2_with_const).fit()
	beta = model.params.iloc[1]

	spread = S1 - (beta * S2)

	# 2. Rolling Z-Score Generation
	rolling_mean = spread.rolling(window=window).mean()
	rolling_std = spread.rolling(window=window).std()
	z_scores = (spread - rolling_mean) / rolling_std

	# 3. State Machine Arrays
	n = len(z_scores)
	s1_positions = np.zeros(n)
	s2_positions = np.zeros(n)

	# State tracking variables: 0 = Flat, 1 = Long, -1 = Short
	current_state = 0

	# 4. Loop
	for i in range(window, n):
		z = z_scores.iloc[i]

		# State: Flat
		if current_state == 0:
			if z > entry_thresh:
				current_state = -1 # Short Spread (Short s1, Long S2)
			elif z < -entry_thresh:
				current_state = 1 # Long Spread (Long S1, Short s2)

		# State: Short the Spread
		elif current_state == -1:
			if z <= exit_thresh:
				current_state = 0 # Reverted to mean, close trade
		# State: Long the Spread
		elif current_state == 1:
			if z >= -exit_thresh:
				current_state = 0 # Reverted to mean, close trade

		# 5. Translation from current state to physical stock share allocation
		if current_state == 1:
			s1_positions[i] = 1.0 # Long 1 share of S1
			s2_positions[i] = -beta # Short 'beta' shares of s2
		elif current_state == -1:
			s1_positions[i] = -1.0 # Short 1 share of s1
			s2_positions[i] = beta # long 'beta' shares of s2
		else:
			s1_positions[i] = 0.0 # flat
			s2_positions[i] = 0.0 # flat

	# put results into dataFrame
	execution_df = pd.DataFrame(index=S1.index)
	execution_df['Z_Score'] = z_scores
	execution_df['S1_Shares'] = s1_positions
	execution_df['S2_Shares'] = s2_positions

	return execution_df, beta

# Step 3: Back Testing
def evaluate_performance(S1, S2, execution_df, initial_capital=100000, stop_loss_thresh=3.5):
    # 1. Extract closing prices and shares
    s1_prices = S1
    s2_prices = S2
    s1_shares = execution_df['S1_Shares']
    s2_shares = execution_df['S2_Shares']
    z_scores = execution_df['Z_Score']

    # 2. Prevent Look-Ahead Bias (The Shift)
    #positions taken today are executed at the close, so they can earn tomorrow's price change
    s1_shares_shifted = s1_shares.shift(1)
    s2_shares_shifted = s2_shares.shift(1)

    # 3. Calculate Daily Dollar Price Changes
    s1_price_diff = s1_prices.diff()
    s2_price_diff = s2_prices.diff()

    # 4. Calculate Daily Strategy PnL (in Dollars)
    daily_pnl = (s1_shares_shifted * s1_price_diff) + (s2_shares_shifted * s2_price_diff)
    daily_pnl = daily_pnl.fillna(0)

    # 5. Risk Control: Hard Statistical Stop-Loss
    is_stopped_out = z_scores.abs() > stop_loss_thresh

    daily_pnl[is_stopped_out.shift(1).fillna(False)] = 0

    # 6. Build the Equity Curve
    equity_curve = initial_capital + daily_pnl.cumsum()
    daily_returns = equity_curve.pct_change().fillna(0)

    # 7. Calculate Performance Stats
    # Sharpe Ratio (Assuming a 0% risk-free rate for draft)
    if daily_returns.std() != 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) #Annualized
    else:
        sharpe_ratio = 0

    # Max Drawdown Calculation
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    # Packaged Summary
    performance_df = pd.DataFrame(index=S1.index)
    performance_df['Equity'] = equity_curve
    performance_df['Drawdown'] = drawdown

    metrics = {
        'Final Value': equity_curve.iloc[-1],
        'Total Return (%)': ((equity_curve.iloc[-1] - initial_capital) / initial_capital) * 100,
        'Annualized Sharp Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100
    }

    return performance_df, metrics

def generate_allocated_execution_states(S1_raw, S2_raw, S1_log, S2_log,
					window=30, entry_thresh=2.0, exit_thresh=0.0,
					initial_capital=100000, allocation_pct=0.10):
    # dynamically sizes share orders based on a percentage of the total portfolio at the exact moment of trade entry.

    # 1. Generate Statistical Signals (using Log Prices)
    S2_with_const = sm.add_constant(S2_log)
    model = sm.OLS(S1_log, S2_with_const).fit()
    beta = model.params.iloc[1]

    spread = S1_log - (beta * S2_log)
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    z_scores = (spread - rolling_mean) / rolling_std
	# 2. Setup Sizing Arrays
    n = len(z_scores)
    s1_shares = np.zeros(n)
    s2_shares = np.zeros(n)

    current_state = 0 # 0 = Flat, 1 = Long, -1 = Short

    # Track physical share blocks locked at entry
    allocated_s1_shares = 0.0
    allocated_s2_shares = 0.0

    # We maintain a mock equity tracker inside the loop to size entries accurately
    running_equity = initial_capital

    # 3. Iterative Exectution Loop
    for i in range(window, n):
        z = z_scores.iloc[i]
        p1 = S1_raw.iloc[i]
        p2 = S2_raw.iloc[i]

        # State Transitions
        if current_state == 0:
	    # Entry Signal triggered
            if abs(z) > entry_thresh:
                current_state = -1 if z > entry_thresh else 1

                # PORTFOLIO ALLOCATION ENGINE
				# Calculate dollar size for this trade slot
                trade_allocation = running_equity * allocation_pct

                # Derive target dollar values per leg using Beta-Neutral formula
                v1_target = trade_allocation / (1 + beta)
                v2_target = v1_target * beta

                # Convert dollar targets into physical share blocks (rounded down)
                allocated_s1_shares = np.floor(v1_target / p1)
                allocated_s2_shares = np.floor(v2_target / p2)
        elif current_state == -1:

            # Exit Signal for short
            if z <= exit_thresh:
                current_state = 0
                allocated_s1_shares = 0.0
                allocated_s2_shares = 0.0

        elif current_state == 1:

            # Exit signal for long
            if z >= -exit_thresh:
                current_state = 0
                allocated_s1_shares = 0.0
                allocated_s2_shares = 0.0

        # 4. Assign current share vectors based on state
            if current_state == 1:
                s1_shares[i] = allocated_s1_shares
                s2_shares[i] = -allocated_s2_shares
            elif current_state == -1:
                s1_shares[i] = -allocated_s1_shares
                s2_shares[i] = allocated_s2_shares
            else:
                s1_shares[i] = 0.0
                s2_shares[i] = 0.0

        # Update our running equity proxy based on yesterday's open positions
        if i > window:
            s1_pnl = s1_shares[i-1] * (S1_raw.iloc[i] - S1_raw.iloc[i-1])
            s2_pnl = s2_shares[i-1] * (S2_raw.iloc[i] - S2_raw.iloc[i-1])
            running_equity += (s1_pnl + s2_pnl)

    # Output DataFrame for the Risk Module
    execution_df = pd.DataFrame(index=S1_raw.index)
    execution_df['Z_Score'] = z_scores
    execution_df['S1_Shares'] = s1_shares
    execution_df['S2_Shares'] = s2_shares

    return execution_df, beta




# Execution

if __name__ == "__main__":
    print("--- STARTING PIPELINE of TESTS ---")

    # 1. Fetching data
    tickers = ['XOM', 'CVX', 'COP', 'EOG','OXY', 'GOOG', 'GOOGL']

    print(f"Downloading historical data for: {len(tickers)} tickers...")
    raw_data = yf.download(tickers, start='2020-01-01', end='2025-01-01', threads=False)['Close'].dropna()
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data = raw_data['Close']
    
    print(f"Initial raw data shape: {raw_data.shape}")

    if raw_data.empty:
        raise ValueError("No data retrieved from Yahoo Finance.")
    
    data = raw_data.ffill().bfill()

    data = data.dropna(axis=1, how='all')
    data = data.dropna(axis=0, how='any')

    print(f"Cleaned data shape: {data.shape}")

    if data.empty:
        raise ValueError("Data is empty after cleaning. Check ticker symbols and date range.")
    
    log_data = np.log(data)

    # 2. Run 5-Stage Filter

    verified_portfolio = {}
    cols = log_data.columns

    print("Beginning Testing...")
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            t1, t2 = cols[i], cols[j]

            passed, metrics = pair_filter(log_data[t1], log_data[t2])
            if passed:
                print(f"\n>>> Validated Pair Found: {t1} vs {t2}")
                print(f" Half-Life: {metrics['Half_Life_Days']:.2f} Days | Hurst: {metrics['Hurst']:.4f}")
                verified_portfolio[(t1, t2)] = metrics

    # 3. Route to Execution

    if len(verified_portfolio) > 0:
        # Pick the first validated pair
        target_pair = list(verified_portfolio.keys())[0]
        asset_1, asset_2 = target_pair
        pair_metrics = verified_portfolio[target_pair]

        # Synce the exection rolloing window to the mathematical Half-Life
        dynamic_window = max(10, int(pair_metrics['Half_Life_Days']))
        print(f"n\Targeting Validated Pair: S1 = {asset_1}, S2 = {asset_2}")
        print(f"Syncing state machine rolling window to empirical Half-Life: {dynamic_window} days")

    else:
        print("\nWarning: No Pairs Survivied the 5-Stage Gauntlet.")
        fall_back_1, fall_back_2 = log_data.columns[0], log_data.columns[1]
        print(f"Forcing baseline fallback test on {fall_back_1} vs {fall_back_2}...")
        asset_1, asset_2 = fall_back_1, fall_back_2
        dynamic_window = 30 # Dafault if forced


    # 4. Signal Generation & Portfolio Allocation
    print("n\Running Modules 2 & 3: Signal Generation & Capital Allocation...")
    execution_output, final_beta = generate_allocated_execution_states(
        data[asset_1], data[asset_2], log_data[asset_1], log_data[asset_2], window=dynamic_window
    )
    print(f"Calculated Hedge Ratio (Beta): {final_beta:.4f}")

    # 5. Risk & Performance Backtest
    print("n\Running Module 4: Risk & Performance Evaluation...")
    perf_df, final_metrics = evaluate_performance(data[asset_1], data[asset_2], execution_output)

    # 6. Output Summary
    print("\n" + "="*40)
    print("    Test Complete: PERFORMANCE sUMMARY    ")
    print("="*40)
    for metrics_name, value in final_metrics.items():
        print(f"{metrics_name:<25}: {value}")
    print("="*40)



