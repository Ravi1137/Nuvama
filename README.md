🛠️ Installation & Setup

# Prerequisites
Python 3.10+ (64 bit)
pip (Python package manager)


# Installation 
Clone this repository
Install dependencies using below command:
pip install pandas numpy matplotlib seaborn

Place the sample_tick_data.csv file in the project root folder where main.py present

# Execution
Run the complete solution:
python main.py


📊 Key Features

**** Data Processing ****
Reads and parses tick-level CSV data
Resamples to 1-minute OHLCV bars
Calculates 20-period moving average and volatility
Stores processed data in efficient Parquet format

**** Trading Strategy ****
Mean-Reversion Logic:
Buy when price < (MA - 1σ)
Sell when price > (MA + 1σ)

**** Risk Controls ****
Maximum position size: 1,000 shares per symbol
Maximum daily loss: 2% of starting capital
End-of-day flat position requirement
Transaction Costs: 0.01% commission per trade

**** Analytics & Reporting ****
Portfolio PnL curve with trade markers
Symbol-specific performance breakdowns
Win rate, Sharpe ratio, and total return calculations
Detailed trade-by-trade execution log
Stress test results with -5% price shock

# Outputs

Processed Data: ohlcv_data.parquet - Cleaned and feature-enhanced data

Visualizations:
pnl_curve.png - Portfolio performance with trade execution points
symbol_performance.png - Comparative analysis across symbols

Console Reports:

Overall performance metrics
Symbol-specific statistics
Detailed trade log with timestamps
Stress test comparison