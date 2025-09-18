# market_data_processor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarketDataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.ohlcv_data = None
        
    def load_data(self):
        """Load and preprocess tick data"""
        self.data = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], format='%d-%m-%y %H:%M')
        return self.data
    
    def resample_ohlcv(self, freq='1min'):
        """Resample tick data to OHLCV bars"""
        if self.data is None:
            self.load_data()
            
        # Group by symbol and resample
        ohlcv_data = self.data.groupby('symbol').resample(freq, on='timestamp').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        # Flatten column multi-index
        ohlcv_data.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_data.reset_index(inplace=True)
        
        # Calculate additional metrics
        ohlcv_data['returns'] = ohlcv_data.groupby('symbol')['close'].pct_change()
        
        self.ohlcv_data = ohlcv_data
        return ohlcv_data
    
    def calculate_indicators(self, window=20):
        """Calculate rolling indicators (MA and volatility)"""
        if self.ohlcv_data is None:
            self.resample_ohlcv()
            
        # Calculate rolling indicators by symbol
        self.ohlcv_data['ma_20'] = self.ohlcv_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=window).mean()
        )
        
        self.ohlcv_data['volatility_20'] = self.ohlcv_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=window).std()
        )
        
        return self.ohlcv_data
    
    def save_to_parquet(self, path='ohlcv_data.parquet'):
        """Save processed data to Parquet format"""
        if self.ohlcv_data is None:
            self.calculate_indicators()
            
        self.ohlcv_data.to_parquet(path, index=False)
        return f"Data saved to {path}"
    
    def view_data_parquet(self, path='ohlcv_data.parquet'):
        bars = pd.read_parquet(path)
        print(bars)

# backtester.py
class Backtester:
    def __init__(self, ohlcv_data, initial_capital=1000000, commission=0.0001):
        self.data = ohlcv_data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.positions = {}
        self.portfolio_history = []
        self.trade_history = []
        self.symbol_performance = {}
        
    def generate_signals(self):
        """Generate trading signals based on mean reversion strategy"""
        # Buy if price < (MA - 1σ), Sell if price > (MA + 1σ)
        self.data['signal'] = 0
        self.data.loc[self.data['close'] < (self.data['ma_20'] - self.data['volatility_20']), 'signal'] = 1  # Buy
        self.data.loc[self.data['close'] > (self.data['ma_20'] + self.data['volatility_20']), 'signal'] = -1  # Sell
        
        return self.data
    
    def apply_risk_management(self, max_position_size=1000, max_daily_loss=0.02):
        """Apply risk management rules"""
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        
    def run_backtest(self):
        """Run the backtest simulation"""
        if 'signal' not in self.data.columns:
            self.generate_signals()
            
        capital = self.initial_capital
        portfolio_value = capital
        daily_capital = capital
        trade_history = []
        
        # Initialize symbol performance tracking
        symbols = self.data['symbol'].unique()
        for symbol in symbols:
            self.symbol_performance[symbol] = {
                'total_pnl': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'total_volume': 0
            }
        
        # Group by date and symbol for daily processing
        self.data['date'] = self.data['timestamp'].dt.date
        dates = sorted(self.data['date'].unique())
        
        for date in dates:
            daily_data = self.data[self.data['date'] == date]
            
            for _, row in daily_data.iterrows():
                symbol = row['symbol']
                
                # Check if we have a signal
                if row['signal'] != 0:
                    # Calculate position size with risk limits
                    position_size = min(self.max_position_size, 
                                       int(capital * 0.1 / row['close']))  # 10% of capital per position
                    
                    if row['signal'] == 1:  # Buy
                        # Check if we already have a position
                        if symbol not in self.positions:
                            cost = position_size * row['close'] * (1 + self.commission)
                            if cost <= capital:
                                self.positions[symbol] = {
                                    'size': position_size,
                                    'entry_price': row['close'],
                                    'entry_time': row['timestamp']
                                }
                                capital -= cost
                                
                                # Log the buy trade
                                trade_history.append({
                                    'timestamp': row['timestamp'],
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'quantity': position_size,
                                    'price': row['close'],
                                    'value': cost,
                                    'commission': position_size * row['close'] * self.commission,
                                    'portfolio_value': portfolio_value
                                })
                                
                    elif row['signal'] == -1 and symbol in self.positions:  # Sell
                        position = self.positions[symbol]
                        revenue = position['size'] * row['close'] * (1 - self.commission)
                        pnl = revenue - (position['size'] * position['entry_price'])
                        capital += revenue
                        
                        # Update symbol performance
                        self.symbol_performance[symbol]['total_pnl'] += pnl
                        self.symbol_performance[symbol]['total_trades'] += 1
                        self.symbol_performance[symbol]['total_volume'] += position['size']
                        if pnl > 0:
                            self.symbol_performance[symbol]['winning_trades'] += 1
                        
                        # Record trade
                        trade_history.append({
                            'timestamp': row['timestamp'],
                            'symbol': symbol,
                            'action': 'SELL',
                            'quantity': position['size'],
                            'price': row['close'],
                            'value': revenue,
                            'commission': position['size'] * row['close'] * self.commission,
                            'pnl': pnl,
                            'return_pct': (pnl / (position['size'] * position['entry_price'])) * 100,
                            'portfolio_value': portfolio_value
                        })
                        
                        del self.positions[symbol]
                
                # Update portfolio value (mark to market)
                current_value = capital
                for pos_symbol, position in self.positions.items():
                    # Find the current price for this position
                    pos_data = daily_data[daily_data['symbol'] == pos_symbol]
                    if not pos_data.empty:
                        current_price = pos_data.iloc[-1]['close']
                        current_value += position['size'] * current_price
                
                portfolio_value = current_value
                
                # Check daily loss limit
                if current_value < daily_capital * (1 - self.max_daily_loss):
                    # Liquidate all positions due to daily loss limit
                    for pos_symbol in list(self.positions.keys()):
                        position = self.positions[pos_symbol]
                        # Find the current price for this symbol
                        pos_data = daily_data[daily_data['symbol'] == pos_symbol]
                        if not pos_data.empty:
                            current_price = pos_data.iloc[-1]['close']
                            revenue = position['size'] * current_price * (1 - self.commission)
                            pnl = revenue - (position['size'] * position['entry_price'])
                            capital += revenue
                            
                            # Update symbol performance
                            self.symbol_performance[pos_symbol]['total_pnl'] += pnl
                            self.symbol_performance[pos_symbol]['total_trades'] += 1
                            self.symbol_performance[pos_symbol]['total_volume'] += position['size']
                            if pnl > 0:
                                self.symbol_performance[pos_symbol]['winning_trades'] += 1
                            
                            trade_history.append({
                                'timestamp': row['timestamp'],
                                'symbol': pos_symbol,
                                'action': 'SELL',
                                'quantity': position['size'],
                                'price': current_price,
                                'value': revenue,
                                'commission': position['size'] * current_price * self.commission,
                                'pnl': pnl,
                                'return_pct': (pnl / (position['size'] * position['entry_price'])) * 100,
                                'reason': 'daily_loss_limit',
                                'portfolio_value': portfolio_value
                            })
                            
                            del self.positions[pos_symbol]
                
                self.portfolio_history.append({
                    'timestamp': row['timestamp'],
                    'portfolio_value': portfolio_value,
                    'capital': capital
                })
            
            # Reset daily capital
            daily_capital = portfolio_value
            
            # Close all positions at market close (end of day)
            for symbol in list(self.positions.keys()):
                position = self.positions[symbol]
                # Find the closing price for this symbol
                closing_data = daily_data[daily_data['symbol'] == symbol]
                if not closing_data.empty:
                    closing_price = closing_data.iloc[-1]['close']
                    revenue = position['size'] * closing_price * (1 - self.commission)
                    pnl = revenue - (position['size'] * position['entry_price'])
                    capital += revenue
                    
                    # Update symbol performance
                    self.symbol_performance[symbol]['total_pnl'] += pnl
                    self.symbol_performance[symbol]['total_trades'] += 1
                    self.symbol_performance[symbol]['total_volume'] += position['size']
                    if pnl > 0:
                        self.symbol_performance[symbol]['winning_trades'] += 1
                    
                    trade_history.append({
                        'timestamp': closing_data.iloc[-1]['timestamp'],
                        'symbol': symbol,
                        'action': 'SELL',
                        'quantity': position['size'],
                        'price': closing_price,
                        'value': revenue,
                        'commission': position['size'] * closing_price * self.commission,
                        'pnl': pnl,
                        'return_pct': (pnl / (position['size'] * position['entry_price'])) * 100,
                        'reason': 'end_of_day',
                        'portfolio_value': portfolio_value
                    })
                    
                    del self.positions[symbol]
        
        # Create results DataFrames
        self.trade_history = pd.DataFrame(trade_history)
        self.portfolio_history = pd.DataFrame(self.portfolio_history)
        
        return self.portfolio_history, self.trade_history
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics from backtest results"""
        if not hasattr(self, 'trade_history') or self.trade_history.empty:
            return {}
        
        # Calculate win rate
        winning_trades = len(self.trade_history[self.trade_history['pnl'] > 0])
        total_trades = len(self.trade_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (annualized)
        portfolio_returns = self.portfolio_history['portfolio_value'].pct_change().dropna()
        if len(portfolio_returns) > 1 and portfolio_returns.std() != 0:
            sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        else:
            sharpe_ratio = 0
        
        # Calculate total return
        total_return = (self.portfolio_history['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Calculate symbol-specific metrics
        symbol_metrics = {}
        for symbol, perf in self.symbol_performance.items():
            if perf['total_trades'] > 0:
                symbol_metrics[symbol] = {
                    'win_rate': perf['winning_trades'] / perf['total_trades'],
                    'total_pnl': perf['total_pnl'],
                    'total_trades': perf['total_trades'],
                    'total_volume': perf['total_volume'],
                    'avg_trade_size': perf['total_volume'] / perf['total_trades'] if perf['total_trades'] > 0 else 0
                }
        
        return {
            'overall': {
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'final_portfolio_value': self.portfolio_history['portfolio_value'].iloc[-1]
            },
            'by_symbol': symbol_metrics
        }
    
    def plot_pnl_curve(self, title='Portfolio PnL Curve'):
        """Plot the portfolio value over time with symbol-specific trades"""
        plt.figure(figsize=(14, 8))
        
        # Plot portfolio value
        plt.plot(self.portfolio_history['timestamp'], 
                self.portfolio_history['portfolio_value'], 
                label='Portfolio Value', linewidth=2.5, color='black')
        
        # Plot trades for each symbol with different colors
        if hasattr(self, 'trade_history') and not self.trade_history.empty:
            colors = {'RELIANCE': 'blue', 'TCS': 'green', 'INFY': 'red'}
            markers = {'BUY': '^', 'SELL': 'v'}
            
            for symbol in self.trade_history['symbol'].unique():
                symbol_trades = self.trade_history[self.trade_history['symbol'] == symbol]
                buy_trades = symbol_trades[symbol_trades['action'] == 'BUY']
                sell_trades = symbol_trades[symbol_trades['action'] == 'SELL']
                
                if not buy_trades.empty:
                    plt.scatter(buy_trades['timestamp'], buy_trades['portfolio_value'],
                               color=colors[symbol], marker=markers['BUY'], s=100,
                               label=f'{symbol} Buy', alpha=0.7)
                
                if not sell_trades.empty:
                    plt.scatter(sell_trades['timestamp'], sell_trades['portfolio_value'],
                               color=colors[symbol], marker=markers['SELL'], s=100,
                               label=f'{symbol} Sell', alpha=0.7)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Portfolio Value (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('pnl_curve.png', dpi=300)
        plt.show()
    
    def plot_symbol_performance(self):
        """Plot performance metrics by symbol"""
        metrics = self.calculate_performance_metrics()
        
        if not metrics or 'by_symbol' not in metrics or not metrics['by_symbol']:
            print("No symbol performance data available.")
            return
        
        # Prepare data for plotting
        symbols = list(metrics['by_symbol'].keys())
        win_rates = [metrics['by_symbol'][s]['win_rate'] for s in symbols]
        pnls = [metrics['by_symbol'][s]['total_pnl'] for s in symbols]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Win rate by symbol
        bars1 = ax1.bar(symbols, win_rates, color=['blue', 'green', 'red'])
        ax1.set_title('Win Rate by Symbol', fontsize=14)
        ax1.set_ylabel('Win Rate', fontsize=12)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2%}', ha='center', va='bottom')
        
        # PnL by symbol
        bars2 = ax2.bar(symbols, pnls, color=['blue', 'green', 'red'])
        ax2.set_title('Total PnL by Symbol', fontsize=14)
        ax2.set_ylabel('PnL (USD)', fontsize=12)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (10 if height >= 0 else -30),
                    f'${height:,.0f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig('symbol_performance.png', dpi=300)
        plt.show()
    
    def print_detailed_trade_report(self):
        """Print detailed trade report"""
        if not hasattr(self, 'trade_history') or self.trade_history.empty:
            print("No trades executed.")
            return
        
        print("\n" + "="*80)
        print("DETAILED TRADE REPORT")
        print("="*80)
        
        # Print overall performance
        metrics = self.calculate_performance_metrics()
        overall = metrics['overall']
        
        print(f"\nOverall Performance:")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Final Portfolio Value: ${overall['final_portfolio_value']:,.2f}")
        print(f"  Total Return: {overall['total_return']:.2f}%")
        print(f"  Total Trades: {overall['total_trades']}")
        print(f"  Winning Trades: {overall['winning_trades']}")
        print(f"  Win Rate: {overall['win_rate']:.2%}")
        print(f"  Sharpe Ratio: {overall['sharpe_ratio']:.2f}")
        
        # Print trades by symbol
        print(f"\nTrades by Symbol:")
        for symbol in self.trade_history['symbol'].unique():
            symbol_trades = self.trade_history[self.trade_history['symbol'] == symbol]
            buy_trades = symbol_trades[symbol_trades['action'] == 'BUY']
            sell_trades = symbol_trades[symbol_trades['action'] == 'SELL']
            
            print(f"\n  {symbol}:")
            print(f"    Total Trades: {len(symbol_trades)} ({len(buy_trades)} buys, {len(sell_trades)} sells)")
            
            if not sell_trades.empty:
                total_pnl = sell_trades['pnl'].sum()
                avg_pnl = sell_trades['pnl'].mean()
                win_rate = len(sell_trades[sell_trades['pnl'] > 0]) / len(sell_trades)
                
                print(f"    Total PnL: ${total_pnl:,.2f}")
                print(f"    Average PnL per Trade: ${avg_pnl:,.2f}")
                print(f"    Win Rate: {win_rate:.2%}")
        
        # Print detailed trade log
        print(f"\nDetailed Trade Log:")
        for _, trade in self.trade_history.iterrows():
            action = trade['action']
            timestamp = trade['timestamp']
            symbol = trade['symbol']
            quantity = trade['quantity']
            price = trade['price']
            
            if action == 'BUY':
                print(f"  {timestamp} - {action} {quantity} shares of {symbol} at ${price:.2f}")
            else:
                pnl = trade['pnl']
                return_pct = trade['return_pct']
                reason = trade.get('reason', 'signal')
                print(f"  {timestamp} - {action} {quantity} shares of {symbol} at ${price:.2f} "
                      f"(PnL: ${pnl:,.2f}, Return: {return_pct:.2f}%, Reason: {reason})")
    
    def stress_test(self, shock=-0.05):
        """Run stress test with price shock"""
        # Create shocked data
        shocked_data = self.data.copy()
        shocked_data['close'] = shocked_data['close'] * (1 + shock)
        shocked_data['open'] = shocked_data['open'] * (1 + shock)
        shocked_data['high'] = shocked_data['high'] * (1 + shock)
        shocked_data['low'] = shocked_data['low'] * (1 + shock)
        
        # Recalculate indicators with shocked prices
        shocked_data['ma_20'] = shocked_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=20).mean()
        )
        
        shocked_data['volatility_20'] = shocked_data.groupby('symbol')['close'].transform(
            lambda x: x.rolling(window=20).std()
        )
        
        # Create new backtester with shocked data
        shocked_backtester = Backtester(shocked_data, self.initial_capital, self.commission)
        shocked_backtester.apply_risk_management(self.max_position_size, self.max_daily_loss)
        shocked_backtester.generate_signals()
        shocked_portfolio, shocked_trades = shocked_backtester.run_backtest()
        shocked_metrics = shocked_backtester.calculate_performance_metrics()
        
        # Plot shocked PnL curve
        shocked_backtester.plot_pnl_curve(title='PnL Curve with -5% Price Shock')
        
        return shocked_metrics, shocked_portfolio, shocked_trades

# main.py
def main():
    # Part 1: Market Data Processing
    print("************************** Part 1: Market Data Processing **************************")
    processor = MarketDataProcessor(r'sample_tick_data.csv')
    data = processor.load_data()
    print(f"Loaded {len(data)} ticks")
    
    ohlcv_data = processor.resample_ohlcv()
    print(f"Resampled to {len(ohlcv_data)} OHLCV bars")
    
    ohlcv_data = processor.calculate_indicators()
    print("Calculated rolling indicators")
    
    processor.save_to_parquet('ohlcv_data.parquet')
    print("Saved data to Parquet format")
    
    print("Viewing Parquet format data")
    processor.view_data_parquet('ohlcv_data.parquet')
    
    # Part 2: Trading Strategy Backtest
    print("\n************************** Part 2 & 3: Trading Strategy Backtest **************************")
    print("\nPart 2: Trading Strategy Backtest")
    backtester = Backtester(ohlcv_data)
    backtester.generate_signals()
    backtester.apply_risk_management(max_position_size=1000, max_daily_loss=0.02)
    
    portfolio_history, trade_history = backtester.run_backtest()
    performance = backtester.calculate_performance_metrics()
    
    print(f"Backtest completed with {performance['overall']['total_trades']} trades")
    print(f"Win Rate: {performance['overall']['win_rate']:.2%}")
    print(f"Sharpe Ratio: {performance['overall']['sharpe_ratio']:.2f}")
    print(f"Total Return: {performance['overall']['total_return']:.2f}%")
    
    # Plot PnL curve
    backtester.plot_pnl_curve()
    
    # Plot symbol performance
    backtester.plot_symbol_performance()
    
    # Print detailed trade report
    backtester.print_detailed_trade_report()
    
    # Part 3: Risk & Stress Analysis
    print("\nPart 3: Risk & Stress Analysis")
    stressed_metrics, stressed_portfolio, stressed_trades = backtester.stress_test(shock=-0.05)
    
    print(f"Stress Test Results (-5% price shock):")
    print(f"Win Rate: {stressed_metrics['overall']['win_rate']:.2%}")
    print(f"Sharpe Ratio: {stressed_metrics['overall']['sharpe_ratio']:.2f}")
    print(f"Total Return: {stressed_metrics['overall']['total_return']:.2f}%")
    
    # Compare results
    return_difference = stressed_metrics['overall']['total_return'] - performance['overall']['total_return']
    print(f"Return Impact: {return_difference:.2f}%")

if __name__ == "__main__":
    main()