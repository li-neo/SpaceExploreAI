import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Union
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from datetime import datetime, timedelta
import requests
import time

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    股票数据采集器 - 从各种来源获取股票数据
    """
    
    def __init__(self, 
                 alpha_vantage_api_key: Optional[str] = None,
                 data_dir: str = "../data/raw"):
        """
        初始化股票数据采集器
        
        参数:
            alpha_vantage_api_key: Alpha Vantage API密钥
            data_dir: 原始数据存储目录
        """
        self.data_dir = data_dir
        self.alpha_vantage_api_key = alpha_vantage_api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化API客户端
        if self.alpha_vantage_api_key:
            self.ts = TimeSeries(key=self.alpha_vantage_api_key, output_format='pandas')
            self.fd = FundamentalData(key=self.alpha_vantage_api_key, output_format='pandas')
        else:
            logger.warning("没有提供Alpha Vantage API密钥，部分功能将不可用")
    
    def get_historical_data_yahoo(self, 
                                  tickers: Union[str, List[str]], 
                                  start_date: str = "2015-01-01",
                                  end_date: Optional[str] = None,
                                  interval: str = "1d",
                                  save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        使用Yahoo Finance API获取历史股票数据
        
        参数:
            tickers: 单个股票代码或股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)，默认为今天
            interval: 数据间隔 ('1d', '1wk', '1mo')
            save: 是否保存到文件
            
        返回:
            包含每个股票代码数据的字典
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        if isinstance(tickers, str):
            tickers = [tickers]
            
        data_dict = {}
        
        for ticker in tickers:
            try:
                logger.info(f"获取{ticker}的历史数据...")
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    logger.warning(f"无法获取{ticker}的数据")
                    continue
                    
                # 重置索引，将日期作为列
                df.reset_index(inplace=True)
                df.rename(columns={"index": "date", "Date": "date"}, inplace=True)
                
                # 确保列名小写
                df.columns = [col.lower() for col in df.columns]
                
                # 添加股票代码列
                df['ticker'] = ticker
                
                # 按日期排序
                df.sort_values('date', inplace=True)
                
                data_dict[ticker] = df
                
                if save:
                    # 创建股票特定的目录
                    ticker_dir = os.path.join(self.data_dir, "price_history", ticker)
                    os.makedirs(ticker_dir, exist_ok=True)
                    
                    # 保存为CSV
                    file_path = os.path.join(ticker_dir, f"{ticker}_{interval}_{start_date}_{end_date}.csv")
                    df.to_csv(file_path, index=False)
                    logger.info(f"已保存{ticker}的数据到{file_path}")
                    
            except Exception as e:
                logger.error(f"获取{ticker}的数据时出错: {str(e)}")
        
        return data_dict
    
    def get_historical_data_alpha_vantage(self, 
                                         ticker: str,
                                         output_size: str = "full",
                                         save: bool = True) -> Optional[pd.DataFrame]:
        """
        使用Alpha Vantage API获取历史股票数据
        
        参数:
            ticker: 股票代码
            output_size: 'compact' (最近100个数据点) 或 'full' (20年数据)
            save: 是否保存到文件
            
        返回:
            包含股票数据的DataFrame
        """
        if not self.alpha_vantage_api_key:
            logger.error("未提供Alpha Vantage API密钥")
            return None
            
        try:
            logger.info(f"通过Alpha Vantage获取{ticker}的历史数据...")
            data, meta_data = self.ts.get_daily(symbol=ticker, outputsize=output_size)
            
            # 重命名列
            data.columns = [col.split(". ")[1].lower() for col in data.columns]
            
            # 重置索引，将日期作为列
            data.reset_index(inplace=True)
            data.rename(columns={"index": "date"}, inplace=True)
            
            # 添加股票代码列
            data['ticker'] = ticker
            
            if save:
                # 创建目录
                ticker_dir = os.path.join(self.data_dir, "price_history", ticker)
                os.makedirs(ticker_dir, exist_ok=True)
                
                # 保存为CSV
                today = datetime.now().strftime("%Y-%m-%d")
                file_path = os.path.join(ticker_dir, f"{ticker}_daily_{today}_alphavantage.csv")
                data.to_csv(file_path, index=False)
                logger.info(f"已保存{ticker}的数据到{file_path}")
                
            return data
            
        except Exception as e:
            logger.error(f"通过Alpha Vantage获取{ticker}的数据时出错: {str(e)}")
            return None
    
    def get_fundamental_data(self, 
                            ticker: str,
                            save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        获取公司基本面数据
        
        参数:
            ticker: 股票代码
            save: 是否保存到文件
            
        返回:
            包含不同基本面数据的字典
        """
        if not self.alpha_vantage_api_key:
            logger.error("未提供Alpha Vantage API密钥")
            return {}
            
        result = {}
        
        try:
            # 获取资产负债表
            logger.info(f"获取{ticker}的资产负债表...")
            balance_sheet, _ = self.fd.get_balance_sheet_annual(symbol=ticker)
            result['balance_sheet'] = balance_sheet
            
            # 获取收入表
            logger.info(f"获取{ticker}的收入表...")
            income_statement, _ = self.fd.get_income_statement_annual(symbol=ticker)
            result['income_statement'] = income_statement
            
            # 获取现金流量表
            logger.info(f"获取{ticker}的现金流量表...")
            cash_flow, _ = self.fd.get_cash_flow_annual(symbol=ticker)
            result['cash_flow'] = cash_flow
            
            # 获取公司概述
            logger.info(f"获取{ticker}的公司概述...")
            overview, _ = self.fd.get_company_overview(symbol=ticker)
            result['overview'] = overview
            
            if save:
                # 创建目录
                fund_dir = os.path.join(self.data_dir, "fundamentals", ticker)
                os.makedirs(fund_dir, exist_ok=True)
                
                # 保存每个数据集
                today = datetime.now().strftime("%Y-%m-%d")
                for data_type, df in result.items():
                    file_path = os.path.join(fund_dir, f"{ticker}_{data_type}_{today}.csv")
                    df.to_csv(file_path)
                    logger.info(f"已保存{ticker}的{data_type}数据到{file_path}")
            
        except Exception as e:
            logger.error(f"获取{ticker}的基本面数据时出错: {str(e)}")
        
        return result
    
    def get_market_indices(self, 
                           indices: List[str] = None,
                           start_date: str = "2015-01-01",
                           end_date: Optional[str] = None,
                           save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        获取市场指数数据
        
        参数:
            indices: 指数列表，例如 ['^GSPC', '^DJI', '^IXIC']
            start_date: 开始日期
            end_date: 结束日期
            save: 是否保存到文件
            
        返回:
            包含每个指数数据的字典
        """
        if indices is None:
            indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX']  # S&P 500, 道琼斯, 纳斯达克, 罗素2000, VIX
            
        return self.get_historical_data_yahoo(indices, start_date, end_date, save=save)
    
    def get_sector_performance(self, save: bool = True) -> Optional[pd.DataFrame]:
        """
        获取各个行业板块的表现数据
        
        参数:
            save: 是否保存到文件
            
        返回:
            包含行业板块表现的DataFrame
        """
        if not self.alpha_vantage_api_key:
            logger.error("未提供Alpha Vantage API密钥")
            return None
            
        try:
            url = f'https://www.alphavantage.co/query?function=SECTOR&apikey={self.alpha_vantage_api_key}'
            r = requests.get(url)
            data = r.json()
            
            # 转换为DataFrame
            all_df = pd.DataFrame()
            for period, sectors in data.items():
                if period == 'Meta Data':
                    continue
                    
                df = pd.DataFrame(list(sectors.items()), columns=['Sector', period])
                
                if all_df.empty:
                    all_df = df
                else:
                    all_df = pd.merge(all_df, df, on='Sector')
            
            if save:
                # 创建目录
                sector_dir = os.path.join(self.data_dir, "sectors")
                os.makedirs(sector_dir, exist_ok=True)
                
                # 保存数据
                today = datetime.now().strftime("%Y-%m-%d")
                file_path = os.path.join(sector_dir, f"sector_performance_{today}.csv")
                all_df.to_csv(file_path, index=False)
                logger.info(f"已保存行业板块表现数据到{file_path}")
                
            return all_df
            
        except Exception as e:
            logger.error(f"获取行业板块表现数据时出错: {str(e)}")
            return None
    
    def get_economic_indicators(self, 
                               indicators: List[str] = None,
                               save: bool = True) -> Dict[str, pd.DataFrame]:
        """
        获取经济指标数据
        
        参数:
            indicators: 经济指标列表，如GDP、CPI等
            save: 是否保存到文件
            
        返回:
            包含每个经济指标数据的字典
        """
        if not self.alpha_vantage_api_key:
            logger.error("未提供Alpha Vantage API密钥")
            return {}
            
        if indicators is None:
            indicators = ['REAL_GDP', 'CPI', 'INFLATION', 'RETAIL_SALES', 'UNEMPLOYMENT', 'NONFARM_PAYROLL']
            
        result = {}
        
        for indicator in indicators:
            try:
                logger.info(f"获取{indicator}经济指标数据...")
                url = f'https://www.alphavantage.co/query?function={indicator}&interval=annual&apikey={self.alpha_vantage_api_key}'
                r = requests.get(url)
                data = r.json()
                
                if 'data' not in data:
                    logger.warning(f"无法获取{indicator}数据，响应: {data}")
                    continue
                    
                df = pd.DataFrame(data['data'])
                result[indicator] = df
                
                if save:
                    # 创建目录
                    econ_dir = os.path.join(self.data_dir, "economic")
                    os.makedirs(econ_dir, exist_ok=True)
                    
                    # 保存数据
                    today = datetime.now().strftime("%Y-%m-%d")
                    file_path = os.path.join(econ_dir, f"{indicator}_{today}.csv")
                    df.to_csv(file_path, index=False)
                    logger.info(f"已保存{indicator}数据到{file_path}")
                    
                # Alpha Vantage API有速率限制，所以需要暂停
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"获取{indicator}数据时出错: {str(e)}")
        
        return result
    
    def get_batch_stock_data(self, 
                            tickers_file: str = None,
                            tickers_list: List[str] = None,
                            start_date: str = "2015-01-01",
                            end_date: Optional[str] = None,
                            include_fundamentals: bool = True,
                            save: bool = True) -> Dict[str, Dict]:
        """
        批量获取多只股票的数据
        
        参数:
            tickers_file: 包含股票代码的文件路径
            tickers_list: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            include_fundamentals: 是否包含基本面数据
            save: 是否保存到文件
            
        返回:
            包含所有股票数据的字典
        """
        # 获取股票列表
        if tickers_file and os.path.exists(tickers_file):
            with open(tickers_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
        elif tickers_list:
            tickers = tickers_list
        else:
            logger.error("未提供股票代码")
            return {}
            
        result = {}
        
        # 获取价格历史数据
        price_data = self.get_historical_data_yahoo(tickers, start_date, end_date, save=save)
        
        for ticker in tickers:
            result[ticker] = {'price_history': price_data.get(ticker)}
            
            # 获取基本面数据
            if include_fundamentals and self.alpha_vantage_api_key:
                # 避免API速率限制
                time.sleep(1)
                fundamental_data = self.get_fundamental_data(ticker, save=save)
                result[ticker]['fundamentals'] = fundamental_data
        
        return result


# 使用示例
if __name__ == "__main__":
    # 获取API密钥
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    
    # 初始化数据采集器
    collector = StockDataCollector(alpha_vantage_api_key=api_key)
    
    # 获取单个股票数据
    aapl_data = collector.get_historical_data_yahoo("AAPL", start_date="2020-01-01")
    
    # 获取多个股票数据
    tickers = ["MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    batch_data = collector.get_batch_stock_data(tickers_list=tickers, start_date="2020-01-01", include_fundamentals=True)
    
    # 获取市场指数
    indices = collector.get_market_indices()
    
    # 获取行业板块表现
    sectors = collector.get_sector_performance()
    
    # 获取经济指标
    econ_data = collector.get_economic_indicators() 