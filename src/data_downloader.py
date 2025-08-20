"""
Data downloader for getting historical data from exchanges
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional
import json

class BinanceDataDownloader:
    """Download historical data from Binance API"""
    
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    TIMEFRAME_MAP = {
        '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
        '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
    }
    
    def __init__(self):
        self.session = requests.Session()
    
    def download_data(self, symbol: str, timeframe: str, 
                     start_date: str, end_date: Optional[str] = None,
                     limit: int = 1000) -> pd.DataFrame:
        """
        Download historical OHLCV data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD' (optional)
            limit: Number of candles per request (max 1000)
            
        Returns:
            DataFrame with OHLCV data
        """
        
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000) if end_date else None
        
        all_data = []
        current_start = start_ts
        
        print(f"Downloading {symbol} {timeframe} data from {start_date}...")
        
        while True:
            params = {
                'symbol': symbol.upper(),
                'interval': self.TIMEFRAME_MAP[timeframe],
                'startTime': current_start,
                'limit': limit
            }
            
            if end_ts:
                params['endTime'] = end_ts
            
            try:
                response = self.session.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                
                all_data.extend(data)
                
                # Update start time for next request
                current_start = data[-1][6] + 1  # Close time + 1ms
                
                # Check if we've reached the end
                if end_ts and current_start >= end_ts:
                    break
                
                print(f"Downloaded {len(all_data)} candles...")
                
                # Respect rate limits
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading data: {e}")
                break
        
        if not all_data:
            raise ValueError("No data downloaded")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'count', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Keep only needed columns and convert types
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        print(f"Downloaded {len(df)} candles successfully!")
        return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to CSV file"""
        df.to_csv(filename, index=False)
        print(f"Data saved to: {filename}")

def download_crypto_data(symbol: str, days: int = 30) -> tuple[str, str]:
    """
    Download crypto data for both 15m and 4h timeframes
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        days: Number of days to download
        
    Returns:
        Tuple of (ltf_filename, htf_filename)
    """
    downloader = BinanceDataDownloader()
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download 15m data
    print("=" * 50)
    df_15m = downloader.download_data(symbol, '15m', start_str, end_str)
    ltf_filename = f"data/{symbol.lower()}_15m_{days}d.csv"
    downloader.save_to_csv(df_15m, ltf_filename)
    
    # Download 4h data
    print("=" * 50)
    df_4h = downloader.download_data(symbol, '4h', start_str, end_str)
    htf_filename = f"data/{symbol.lower()}_4h_{days}d.csv"
    downloader.save_to_csv(df_4h, htf_filename)
    
    return ltf_filename, htf_filename

async def download_specific_timeframe(symbol: str, timeframe: str, days: int, 
                                    end_date: Optional[datetime] = None) -> str:
    """
    Download data for a specific timeframe
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        timeframe: Timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w')
        days: Number of days to download
        end_date: End date (defaults to now)
        
    Returns:
        Filename of the downloaded data
    """
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=days)
    
    downloader = BinanceDataDownloader()
    
    # Download data
    df = downloader.download_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Save to file
    filename = f"{symbol.lower()}_{timeframe}_{days}d.csv"
    filepath = f"data/{filename}"
    downloader.save_to_csv(df, filepath)
    
    return filename

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download crypto data from Binance')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--days', type=int, default=30, help='Days to download (default: 30)')
    parser.add_argument('--timeframe', help='Single timeframe to download (e.g., 15m, 4h)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.timeframe and args.start:
        # Single timeframe download
        downloader = BinanceDataDownloader()
        df = downloader.download_data(args.symbol, args.timeframe, args.start, args.end)
        filename = f"data/{args.symbol.lower()}_{args.timeframe}_{args.days}d.csv"
        downloader.save_to_csv(df, filename)
    else:
        # Download both timeframes
        ltf_file, htf_file = download_crypto_data(args.symbol, args.days)
        print(f"\n✅ Ready to use:")
        print(f"LTF: {ltf_file}")
        print(f"HTF: {htf_file}")
        print(f"\n🚀 Run bot:")
        print(f"venv\\Scripts\\python.exe main.py --ltf {ltf_file} --htf {htf_file}")
