import os
import sys
import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from log.logger import init_logger

# # 设置日志

logger = init_logger("download_data.log")


def download_yahoo_finance_data(
    ticker: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    interval: str
) -> bool:
    """
    从Yahoo Finance下载股票数据
    
    参数:
        ticker: 股票代码
        start_date: 开始日期，格式为'YYYY-MM-DD'
        end_date: 结束日期，格式为'YYYY-MM-DD'
        output_dir: 输出目录
        
    返回:
        是否成功下载
    """
    try:
        logger.info(f"从Yahoo Finance下载股票数据: {ticker}")
        
        # 下载数据
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, interval=interval)
        
        # 确保下载成功
        if data.empty:
            logger.error(f"下载失败，数据为空: {ticker}")
            return False
            
        # 重置索引，将日期列从索引转为普通列
        data.reset_index(inplace=True)
        
        # 添加股票代码列
        data['ticker'] = ticker
        
        # 创建保存目录
        ticker_dir = os.path.join(output_dir, "price_history", ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        
        # 构建输出文件名
        now = datetime.now().strftime("%Y%m%d")
        output_file = os.path.join(ticker_dir, f"{ticker}_yahoo_{now}.csv")
        
        # 保存数据
        data.to_csv(output_file, index=False)
        logger.info(f"成功保存数据到: {output_file}，共 {len(data)} 行")
        
        return True
    except Exception as e:
        logger.error(f"下载股票数据时出错: {ticker}, 错误: {str(e)}")
        return False


def download_stocks(args):
    """
    下载多只股票的数据
    
    参数:
        args: 命令行参数
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析股票代码列表
    tickers = args.tickers.split(',')
    logger.info(f"将下载 {len(tickers)} 只股票的数据: {args.tickers}")
    
    # 下载每只股票的数据
    success_count = 0
    for ticker in tqdm(tickers, desc="下载股票数据"):
        if args.source.lower() == 'yahoo':
            if download_yahoo_finance_data(ticker, args.start_date, args.end_date, args.output_dir, args.interval):
                success_count += 1
        else:
            logger.error(f"不支持的数据源: {args.source}")
            
    logger.info(f"完成下载，成功: {success_count}/{len(tickers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载股票历史价格数据")
    
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOG,AMZN,TSLA,NVDA,QQQ", help="股票代码，多个用逗号分隔")
    parser.add_argument("--start_date", type=str, default="2010-03-11", help="开始日期，格式为'YYYY-MM-DD'")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="结束日期，格式为'YYYY-MM-DD'")
    parser.add_argument("--output_dir", type=str, default="../data/raw", help="输出目录")
    parser.add_argument("--source", type=str, default="yahoo", help="数据源，如'yahoo'或'alphavantage'")
    parser.add_argument("--interval", type=str, default="1d", help="数据间隔，如'1m'或'1d'")
    args = parser.parse_args()
    
    #下载股票数据
    download_stocks(args) 