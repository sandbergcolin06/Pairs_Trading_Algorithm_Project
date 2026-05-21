import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

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
        'Total Return (%)': ((equity_curve.iloc[-1] - initial_capital) / initial_capital),
        'Annualized Sharp Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100
    }

    return performance_df, metrics


# Execution

if __name__ == "__main__":
    print("--- STARTING PIPELINE SMOKE TEST ---")

    # 1. Fetching data
    tickers = ['XOM', 'CVX']
    print(f"Downloading historical data for: {tickers}")
    raw_data = yf.download(tickers, start='2020-01-01', end='2025-01-01', threads=False)['Close']
    data = raw_data.dropna()
    log_data = np.log(data)
    print("Running Module 1: Cointegration Scan...")
    _,_, identified_pairs = find_cointegrated_pairs(log_data)
    print(f"Pairs flagged as cointegrated: {identified_pairs}")
    if len(identified_pairs) > 0:
	    asset_1, asset_2 = identified_pairs[0]
    else:
        print("Warning: No pairs beat the .05 threshold.")
        asset_1, asset_2 = tickers
    print(f"Targeting Pairs: S1 = {asset_1}, S2 = {asset_2}")
    # run signal generation
    print("Run Signal Generation")
    execution_output, final_beta = generate_execution_states(log_data[asset_1], log_data[asset_2])
    print(f"Calculated Hedge Ratio (Beta): {final_beta: .4f}")
    # run backtesting
    print("Running Back-Testing")
    perf_df, final_metrics = evaluate_performance(data[asset_1], data[asset_2], execution_output)
    print("\n--- Test Complete: PERFORMANCE SUMMARY ---")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name:<25}: {value}")




