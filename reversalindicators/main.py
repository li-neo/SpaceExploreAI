#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
趋势反转策略实时监控和回测
-------------------------------
此脚本实时监控美国十大巨头股票的趋势反转信号，
每分钟扫描一次分钟级K线数据，当信号强度超过80%时发出警告。
同时支持历史数据回测功能。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import time
import schedule
import traceback
import warnings
import json
from typing import List, Dict, Tuple, Optional

# 忽略警告
warnings.filterwarnings("ignore")

# 将父目录添加到路径以导入计算器
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reversalindicators.trend_reversal_calculator import TrendReversalSignal
from reversalindicators.trend_args import TrendArgs

# 尝试导入日志模块，如果不存在则使用标准输出
try:
    from log.logger import get_logger
    logger = get_logger(__name__, "trend_reversal")
    use_logger = True
except ImportError:
    use_logger = False
    print("日志模块不可用，使用标准输出")

# 美国十大巨头股票列表
US_TOP_10 = [
    "AAPL",  # 苹果
    "MSFT",  # 微软
    "GOOGL", # 谷歌
    "AMZN",  # 亚马逊
    "META",  # Meta(Facebook)
    "TSLA",  # 特斯拉
    "NVDA",  # 英伟达
    "BRK-B", # 伯克希尔哈撒韦
    "JPM",   # 摩根大通
    "V"      # Visa
]

def safe_iloc(df, start, end=None):
    """
    安全地使用iloc，避免numpy数组和pandas DataFrame的冲突。
    
    参数:
    -----------
    df : pandas.DataFrame
        需要切片的DataFrame
    start : int
        起始位置
    end : int, optional
        结束位置
        
    返回:
    --------
    pandas.DataFrame 或 pandas.Series : 切片结果
    """
    try:
        if end is not None:
            return df.iloc[start:end+1]
        else:
            return df.iloc[start]
    except Exception as e:
        print(f"iloc操作错误: {str(e)}")
        # 尝试转换为pandas Series后再操作
        if isinstance(df, np.ndarray):
            df_series = pd.Series(df)
            if end is not None:
                return df_series.iloc[start:end+1]
            else:
                return df_series.iloc[start]
        raise e

def process_dataframe(df, ticker):
    """
    统一处理DataFrame的格式和列名。
    
    参数:
    -----------
    df : pandas.DataFrame
        需要处理的DataFrame
    ticker : str
        股票代码
        
    返回:
    --------
    pandas.DataFrame : 处理后的DataFrame
    """
    try:
        # 处理多级索引列名的情况
        if isinstance(df.columns, pd.MultiIndex):
            new_df = pd.DataFrame(index=df.index)
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if (col, ticker) in df.columns:
                    new_df[col.lower()] = df[(col, ticker)].copy()
                else:
                    raise KeyError(f"在多级索引中未找到列 ({col}, {ticker})")
            df = new_df
        else:
            # 如果是普通列名，则转换为小写
            df.columns = [col.lower() for col in df.columns]
        
        # 确保必要的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"数据中缺少必要的列: {missing_columns}")
        
        # 确保数据类型正确
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 确保索引是有序的
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        return df
        
    except Exception as e:
        raise Exception(f"处理DataFrame时出错: {str(e)}")

def patch_trend_reversal_signal(sensitivity=1.0):
    """
    修补TrendReversalSignal类中的潜在问题，并调整背离检测的敏感度
    
    参数:
    -----------
    sensitivity : float
        背离检测的敏感度，值越大越容易检测到背离
        1.0 表示正常敏感度，2.0 表示高敏感度
    """
    # 保存原始方法
    original_analyze_divergence = TrendReversalSignal.analyze_divergence
    original_analyze_volume = TrendReversalSignal.analyze_volume
    original_analyze_support_resistance = TrendReversalSignal.analyze_support_resistance
    
    # 修补analyze_divergence方法
    def patched_analyze_divergence(self, lookback=20):
        """
        分析价格-指标背离，支持调整敏感度。
        
        参数:
        -----------
        lookback : int
            用于背离分析的回溯周期数
            
        返回:
        --------
        int : 背离得分 (0-4)
        """
        # 获取最近数据进行分析
        recent_data = self.data.tail(lookback).copy()
        
        # 寻找局部价格高点和低点
        recent_data['price_high'] = recent_data['close'].rolling(5, center=True).apply(
            lambda x: 1 if x[2] == max(x) else 0, raw=True)
        recent_data['price_low'] = recent_data['close'].rolling(5, center=True).apply(
            lambda x: 1 if x[2] == min(x) else 0, raw=True)
        
        # 寻找指标高点和低点
        for indicator in ['rsi', 'macd', 'slowk']:
            recent_data[f'{indicator}_high'] = recent_data[indicator].rolling(5, center=True).apply(
                lambda x: 1 if x[2] == max(x) else 0, raw=True)
            recent_data[f'{indicator}_low'] = recent_data[indicator].rolling(5, center=True).apply(
                lambda x: 1 if x[2] == min(x) else 0, raw=True)
        
        # 删除NaN值
        recent_data = recent_data.dropna()
        
        # 寻找潜在的背离点
        divergence_score = 0
        divergence_indicators = []
        
        # 检查顶背离（价格创新高但指标未创新高）
        price_highs = recent_data[recent_data['price_high'] == 1].index
        if len(price_highs) >= 2:
            last_two_highs = price_highs[-2:]
            price_change = (recent_data.loc[last_two_highs[-1], 'close'] / 
                           recent_data.loc[last_two_highs[-2], 'close'] - 1) * 100
            
            # 检查价格是否创新高 - 调整敏感度
            # 原始条件: price_change > 0
            # 调整后: 允许轻微下降但仍视为"创新高"
            if price_change > -1.0 * sensitivity:  # 允许轻微下降
                bearish_divs = []
                for indicator in ['rsi', 'macd', 'slowk']:
                    ind_change = (recent_data.loc[last_two_highs[-1], indicator] / 
                                 recent_data.loc[last_two_highs[-2], indicator] - 1) * 100
                    # 如果指标没有创新高，则为顶背离 - 调整敏感度
                    # 原始条件: ind_change < 0
                    # 调整后: 即使指标略微上升也可能视为背离
                    if ind_change < 1.0 / sensitivity:  # 允许轻微上升
                        bearish_divs.append(indicator)
                
                if len(bearish_divs) == 1:
                    divergence_score += 1
                elif len(bearish_divs) == 2:
                    divergence_score += 2
                elif len(bearish_divs) >= 3:
                    divergence_score += 3
                
                # 检查三重背离
                if len(price_highs) >= 3:
                    triple_div = True
                    for indicator in bearish_divs:
                        ind_vals = [recent_data.loc[ph, indicator] for ph in price_highs[-3:]]
                        # 调整敏感度
                        if not (ind_vals[0] > ind_vals[1] * (1 - 0.05/sensitivity) and 
                                ind_vals[1] > ind_vals[2] * (1 - 0.05/sensitivity)):
                            triple_div = False
                    
                    if triple_div and len(bearish_divs) > 0:
                        divergence_score = 3  # 确保三重背离获得高分
                
                # 检查背离幅度 - 调整敏感度
                # 原始条件: abs(price_change) > 20
                if bearish_divs and abs(price_change) > 20 / sensitivity:
                    divergence_score += 1
                
                if bearish_divs:
                    self.signal_type = "PUT"
                    divergence_indicators = bearish_divs
        
        # 检查底背离（价格创新低但指标未创新低）
        price_lows = recent_data[recent_data['price_low'] == 1].index
        if len(price_lows) >= 2 and divergence_score == 0:  # 仅在未发现顶背离时检查
            last_two_lows = price_lows[-2:]
            price_change = (recent_data.loc[last_two_lows[-1], 'close'] / 
                           recent_data.loc[last_two_lows[-2], 'close'] - 1) * 100
            
            # 检查价格是否创新低 - 调整敏感度
            # 原始条件: price_change < 0
            # 调整后: 允许轻微上升但仍视为"创新低"
            if price_change < 1.0 * sensitivity:  # 允许轻微上升
                bullish_divs = []
                for indicator in ['rsi', 'macd', 'slowk']:
                    ind_change = (recent_data.loc[last_two_lows[-1], indicator] / 
                                 recent_data.loc[last_two_lows[-2], indicator] - 1) * 100
                    # 如果指标没有创新低，则为底背离 - 调整敏感度
                    # 原始条件: ind_change > 0
                    # 调整后: 即使指标略微下降也可能视为背离
                    if ind_change > -1.0 / sensitivity:  # 允许轻微下降
                        bullish_divs.append(indicator)
                
                if len(bullish_divs) == 1:
                    divergence_score += 1
                elif len(bullish_divs) == 2:
                    divergence_score += 2
                elif len(bullish_divs) >= 3:
                    divergence_score += 3
                
                # 检查三重背离
                if len(price_lows) >= 3:
                    triple_div = True
                    for indicator in bullish_divs:
                        ind_vals = [recent_data.loc[pl, indicator] for pl in price_lows[-3:]]
                        # 调整敏感度
                        if not (ind_vals[0] < ind_vals[1] * (1 + 0.05/sensitivity) and 
                                ind_vals[1] < ind_vals[2] * (1 + 0.05/sensitivity)):
                            triple_div = False
                    
                    if triple_div and len(bullish_divs) > 0:
                        divergence_score = 3  # 确保三重背离获得高分
                
                # 检查背离幅度 - 调整敏感度
                # 原始条件: abs(price_change) > 20
                if bullish_divs and abs(price_change) > 20 / sensitivity:
                    divergence_score += 1
                
                if bullish_divs:
                    self.signal_type = "CALL"
                    divergence_indicators = bullish_divs
        
        # 将分数上限设为4
        divergence_score = min(divergence_score, 4)
        
        self.signal_details["divergence"] = divergence_score
        self.divergence_indicators = divergence_indicators
        
        return divergence_score
    
    # 修补analyze_volume方法
    def patched_analyze_volume(self, pivot_bars=5, lookback=30):
        try:
            if self.signal_type is None:
                return 0
            
            recent_data = self.data.tail(lookback).copy()
            
            # 确保数据足够
            if len(recent_data) < 10:
                return 0
            
            # 根据信号类型识别潜在的枢轴点
            if self.signal_type == "PUT":
                # 对于PUT信号，寻找最近的高点
                pivot_idx = recent_data['close'].idxmax()
            else:  # CALL
                # 对于CALL信号，寻找最近的低点
                pivot_idx = recent_data['close'].idxmin()
            
            # 获取枢轴点在数据框中的位置
            try:
                pivot_pos = recent_data.index.get_loc(pivot_idx)
            except:
                return 0
            
            # 确保枢轴点前后有足够的数据
            if pivot_pos < pivot_bars or pivot_pos >= len(recent_data) - pivot_bars:
                return 0
            
            # 使用安全的iloc操作
            pivot_start = max(0, pivot_pos - pivot_bars)
            pivot_end = min(len(recent_data) - 1, pivot_pos + pivot_bars)
            
            pivot_volume = safe_iloc(recent_data, pivot_start, pivot_end)['volume'].mean()
            
            # 获取基准成交量（枢轴点之前）
            baseline_start = max(0, pivot_start - 20)
            baseline_end = pivot_start - 1
            
            if baseline_end < baseline_start:
                return 0
            
            baseline_volume = safe_iloc(recent_data, baseline_start, baseline_end)['volume'].mean()
            
            # 计算成交量比率
            if baseline_volume == 0:
                return 0
            
            volume_ratio = pivot_volume / baseline_volume
            
            # 根据成交量比率评分
            volume_score = 0
            if volume_ratio >= 3 and volume_ratio < 4:
                volume_score = 1
            elif volume_ratio >= 4 and volume_ratio < 5:
                volume_score = 2
            elif volume_ratio >= 5:
                volume_score = 3
            
            # 检查成交量放大是否持续多根K线
            high_vol_bars = sum(safe_iloc(recent_data, pivot_start, pivot_end)['volume'] > 
                               2 * baseline_volume)
            
            if high_vol_bars >= 3:
                volume_score += 1
            
            # 将分数上限设为4
            volume_score = min(volume_score, 4)
            
            self.signal_details["volume"] = volume_score
            self.volume_ratio = volume_ratio
            
            return volume_score
        except Exception as e:
            print(f"分析成交量时出错: {str(e)}")
            return 0
    
    # 修补analyze_support_resistance方法
    def patched_analyze_support_resistance(self, lookback=100):
        try:
            if self.signal_type is None:
                return 0
            
            # 确保数据足够
            if len(self.data) < lookback:
                lookback = len(self.data) - 1
            
            if lookback < 10:
                return 0
            
            hist_data = self.data.tail(lookback).copy()
            current_price = self.data['close'].iloc[-1]
            
            # 寻找重要价格水平
            sr_levels = []
            
            # 1. 整数关口
            if current_price > 0:
                price_magnitude = 10 ** int(np.log10(current_price))
                round_levels = [
                    int(current_price / price_magnitude) * price_magnitude,
                    int(current_price / (price_magnitude/2)) * (price_magnitude/2),
                    int(current_price / (price_magnitude/10)) * (price_magnitude/10)
                ]
                sr_levels.extend(round_levels)
            
            # 2. 前期高低点
            try:
                hist_data['price_high'] = hist_data['high'].rolling(10, center=True).apply(
                    lambda x: x[5] if len(x) > 5 and x[5] == max(x) else np.nan, raw=True)
                hist_data['price_low'] = hist_data['low'].rolling(10, center=True).apply(
                    lambda x: x[5] if len(x) > 5 and x[5] == min(x) else np.nan, raw=True)
                
                highs = hist_data['price_high'].dropna().tolist()
                lows = hist_data['price_low'].dropna().tolist()
                
                sr_levels.extend(highs + lows)
            except Exception as e:
                print(f"计算高低点时出错: {str(e)}")
            
            # 3. 移动平均线
            try:
                latest = self.data.iloc[-1]
                ma_levels = []
                for ma in ['sma20', 'sma50', 'sma200']:
                    if ma in latest and not pd.isna(latest[ma]):
                        ma_levels.append(latest[ma])
                sr_levels.extend(ma_levels)
            except Exception as e:
                print(f"获取移动平均线时出错: {str(e)}")
            
            # 寻找距离当前价格最近的水平
            sr_levels = [level for level in sr_levels if not np.isnan(level)]
            if not sr_levels:
                return 0
                
            sr_levels.sort()
            
            # 寻找在当前价格5%范围内的水平
            nearby_levels = [level for level in sr_levels 
                            if abs(level/current_price - 1) < 0.05]
            
            # 计算重叠水平（彼此相差1%以内）
            clusters = []
            for level in nearby_levels:
                added = False
                for cluster in clusters:
                    if any(abs(level/l - 1) < 0.01 for l in cluster):
                        cluster.append(level)
                        added = True
                        break
                if not added:
                    clusters.append([level])
            
            # 根据支撑/阻力强度评分
            sr_score = 0
            
            # 基于附近水平数量的基本分数
            if len(nearby_levels) == 1:
                sr_score = 1
            elif len(nearby_levels) >= 2:
                sr_score = 2
            
            # 聚集水平的额外分数
            if any(len(cluster) >= 2 for cluster in clusters):
                sr_score += 1
            
            # 将分数上限设为3
            sr_score = min(sr_score, 3)
            
            self.signal_details["support_resistance"] = sr_score
            self.sr_levels = nearby_levels
            
            return sr_score
        except Exception as e:
            print(f"分析支撑阻力位时出错: {str(e)}")
            return 0
    
    # 替换原始方法
    TrendReversalSignal.analyze_divergence = patched_analyze_divergence
    TrendReversalSignal.analyze_volume = patched_analyze_volume
    TrendReversalSignal.analyze_support_resistance = patched_analyze_support_resistance

def analyze_stock(ticker, ref_ticker="QQQ", period="1d", interval="1m", timeframes=None, verbose=True, debug_mode=False):
    """
    分析股票的趋势反转信号。
    
    参数:
    -----------
    ticker : str
        要分析的股票代码
    ref_ticker : str
        用于相对强弱比较的参考股票代码
    period : str
        下载数据的时间段（例如，"1d"，"5d"）
    interval : str
        数据间隔（例如，"1m"，"5m"）
    timeframes : list, optional
        要分析的多个时间周期列表，例如 [("5m", "1d"), ("1h", "5d")]
        每个元素是一个元组 (interval, period)
    verbose : bool
        是否打印详细信息
    debug_mode : bool
        是否启用调试模式，提供更详细的分析信息
        
    返回:
    --------
    dict : 信号分析结果
    """
    if verbose:
        print(f"\n分析 {ticker} 的趋势反转信号...")
    
    # 如果指定了多个时间周期，则分析每个时间周期
    if timeframes:
        all_results = []
        for tf_interval, tf_period in timeframes:
            if verbose:
                print(f"\n分析 {ticker} 在 {tf_interval} 时间周期上的信号 (数据周期: {tf_period})...")
            result = analyze_single_timeframe(ticker, ref_ticker, tf_period, tf_interval, verbose, debug_mode)
            if result and result['signal_type'] is not None:
                all_results.append(result)
                
        # 如果有多个时间周期的信号，选择信号强度最高的一个
        if all_results:
            strongest_result = max(all_results, key=lambda x: x['signal_strength'])
            if verbose:
                print(f"\n在所有时间周期中，{ticker} 的最强信号出现在 {strongest_result.get('timeframe', '未知')} 时间周期")
            return strongest_result
        else:
            if verbose:
                print(f"\n{ticker} 在所有时间周期上都没有检测到信号")
            return None
    
    # 如果没有指定多个时间周期，则只分析单个时间周期
    return analyze_single_timeframe(ticker, ref_ticker, period, interval, verbose, debug_mode)

def analyze_single_timeframe(ticker, ref_ticker="QQQ", period="1d", interval="1m", verbose=True, debug_mode=False):
    """
    在单个时间周期上分析股票的趋势反转信号。
    
    参数:
    -----------
    ticker : str
        要分析的股票代码
    ref_ticker : str
        用于相对强弱比较的参考股票代码
    period : str
        下载数据的时间段（例如，"1d"，"5d"）
    interval : str
        数据间隔（例如，"1m"，"5m"）
    verbose : bool
        是否打印详细信息
    debug_mode : bool
        是否启用调试模式，提供更详细的分析信息
        
    返回:
    --------
    dict : 信号分析结果
    """
    try:
        # 下载数据
        stock_data = yf.download(ticker, period=period, interval=interval, progress=False)
        ref_data = yf.download(ref_ticker, period=period, interval=interval, progress=False)
        
        if verbose:
            print(f"下载了 {len(stock_data)} 条 {ticker} 数据记录 (间隔: {interval})")
            print(f"下载了 {len(ref_data)} 条 {ref_ticker} 数据记录 (间隔: {interval})")
        
        if stock_data.empty or ref_data.empty:
            if verbose:
                print(f"错误: 无法下载 {ticker} 或 {ref_ticker} 的数据")
            return None
            
        # 检查是否有足够的数据点进行分析
        min_data_points = 30
        if interval == "1h":
            min_data_points = 24  # 至少需要24小时的数据
        elif interval == "5m":
            min_data_points = 100  # 至少需要100个5分钟K线
            
        if len(stock_data) < min_data_points:
            if verbose:
                print(f"警告: {ticker} 的数据点不足 ({len(stock_data)}), 需要至少{min_data_points}个数据点")
            return None
        
        # 统一处理DataFrame
        stock_data = process_dataframe(stock_data, ticker)
        ref_data = process_dataframe(ref_data, ref_ticker)
        
        # 初始化信号计算器
        signal = TrendReversalSignal(ticker, reference_ticker=ref_ticker)
        signal.load_data(stock_data, ref_data)
        
        # 如果是调试模式，分析每个组成部分并打印详细信息
        if debug_mode:
            print(f"\n===== {ticker} ({interval}) 调试信息 =====")
            # 分析背离
            div_score = signal.analyze_divergence()
            print(f"背离分析结果: {div_score}/4")
            if hasattr(signal, 'divergence_indicators'):
                print(f"背离指标: {signal.divergence_indicators}")
            print(f"信号类型: {signal.signal_type}")
            
            # 如果没有检测到背离，则不会有信号类型
            if signal.signal_type is None:
                print("未检测到背离，因此没有产生信号")
                # 检查是否有潜在的背离
                recent_data = stock_data.tail(20).copy()
                print(f"最近20个K线的价格变化: {recent_data['close'].pct_change().sum()*100:.2f}%")
                
                # 检查技术指标
                if 'rsi' in recent_data.columns:
                    print(f"RSI范围: {recent_data['rsi'].min():.2f} - {recent_data['rsi'].max():.2f}")
                if 'macd' in recent_data.columns:
                    print(f"MACD范围: {recent_data['macd'].min():.2f} - {recent_data['macd'].max():.2f}")
                if 'slowk' in recent_data.columns:
                    print(f"Stochastic K范围: {recent_data['slowk'].min():.2f} - {recent_data['slowk'].max():.2f}")
                
                # 检查价格高低点
                print(f"检查价格高低点...")
                try:
                    recent_data['price_high'] = recent_data['close'].rolling(5, center=True).apply(
                        lambda x: 1 if x[2] == max(x) else 0, raw=True)
                    recent_data['price_low'] = recent_data['close'].rolling(5, center=True).apply(
                        lambda x: 1 if x[2] == min(x) else 0, raw=True)
                    
                    price_highs = recent_data[recent_data['price_high'] == 1].index
                    price_lows = recent_data[recent_data['price_low'] == 1].index
                    
                    print(f"检测到 {len(price_highs)} 个价格高点和 {len(price_lows)} 个价格低点")
                    
                    if len(price_highs) >= 2:
                        last_two_highs = price_highs[-2:]
                        price_change = (recent_data.loc[last_two_highs[-1], 'close'] / 
                                      recent_data.loc[last_two_highs[-2], 'close'] - 1) * 100
                        print(f"最近两个高点的价格变化: {price_change:.2f}%")
                    
                    if len(price_lows) >= 2:
                        last_two_lows = price_lows[-2:]
                        price_change = (recent_data.loc[last_two_lows[-1], 'close'] / 
                                      recent_data.loc[last_two_lows[-2], 'close'] - 1) * 100
                        print(f"最近两个低点的价格变化: {price_change:.2f}%")
                except Exception as e:
                    print(f"检查价格高低点时出错: {str(e)}")
            else:
                # 如果有信号类型，继续分析其他组成部分
                vol_score = signal.analyze_volume()
                print(f"成交量分析结果: {vol_score}/4")
                if hasattr(signal, 'volume_ratio'):
                    print(f"成交量比率: {signal.volume_ratio:.2f}")
                
                rs_score = signal.analyze_relative_strength()
                print(f"相对强弱分析结果: {rs_score}/3")
                if hasattr(signal, 'relative_strength'):
                    print(f"相对强弱比率: {signal.relative_strength:.2f}")
                
                sr_score = signal.analyze_support_resistance()
                print(f"支撑/阻力分析结果: {sr_score}/3")
                if hasattr(signal, 'sr_levels'):
                    print(f"支撑/阻力位: {signal.sr_levels}")
                
                # 计算总分
                total_score = div_score + vol_score + rs_score + sr_score
                print(f"总分: {total_score}/14")
            
            print("="*30)
        
        # 计算信号强度
        result = signal.calculate_signal_strength()
        
        # 添加时间周期信息
        if result and result['signal_type'] is not None:
            result['timeframe'] = interval
            result['data_period'] = period
        
        # 如果需要详细输出
        if verbose and result and result['signal_type'] is not None:
            print(f"\n=== {ticker} ({interval}) 趋势反转信号分析 ===")
            print(f"股票代码: {result['ticker']}")
            print(f"信号类型: {result['signal_type']}")
            print(f"信号强度: {result['signal_strength']}/14")
            print(f"信号等级: {result['signal_class']}")
            print(f"成功概率: {result['success_probability']}")
            print(f"建议仓位大小: {result['recommended_position']}")
            print("\n详细得分:")
            print(f"  背离得分: {result['details']['divergence_score']}/4")
            print(f"  成交量得分: {result['details']['volume_score']}/4")
            print(f"  相对强弱得分: {result['details']['relative_strength_score']}/3")
            print(f"  支撑/阻力得分: {result['details']['support_resistance_score']}/3")
            
            # 绘制信号
            try:
                signal.plot_signal()
            except Exception as e:
                print(f"绘制图表时出错: {str(e)}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"分析 {ticker} ({interval}) 时出错: {str(e)}")
            print(traceback.format_exc())
        return None

def debug_no_signal_stocks(tickers=["GOOGL", "NVDA", "V"], ref_ticker="QQQ", timeframes=None, sensitivity=1.0):
    """
    调试为什么某些股票没有产生信号。
    
    参数:
    -----------
    tickers : list
        要调试的股票代码列表
    ref_ticker : str
        用于相对强弱比较的参考股票代码
    timeframes : list, optional
        要分析的多个时间周期列表，例如 [("5m", "1d"), ("1h", "5d")]
    sensitivity : float
        背离检测的敏感度，值越大越容易检测到背离
    """
    print("\n===== 开始调试无信号股票 =====")
    print(f"使用敏感度: {sensitivity} (值越大越容易检测到信号)")
    
    # 应用补丁
    patch_trend_reversal_signal(sensitivity)
    
    # 如果没有指定时间周期，使用默认的多个时间周期
    if timeframes is None:
        timeframes = [
            ("1m", "1d"),   # 1分钟K线，1天数据
            ("5m", "1d"),   # 5分钟K线，1天数据
            ("15m", "5d"),  # 15分钟K线，5天数据
            ("1h", "30d")   # 1小时K线，30天数据
        ]
    
    for ticker in tickers:
        print(f"\n调试 {ticker} 为什么没有产生信号...")
        try:
            # 使用调试模式分析股票
            analyze_stock(ticker, ref_ticker, timeframes=timeframes, verbose=True, debug_mode=True)
        except Exception as e:
            print(f"调试 {ticker} 时出错: {str(e)}")
            print(traceback.format_exc())
    
    print("\n===== 调试完成 =====")

def scan_top_stocks(tickers=US_TOP_10, ref_ticker="QQQ", timeframes=None, warning_threshold=11, debug_no_signal=False):
    """
    扫描美国十大巨头股票，寻找强烈的趋势反转信号。
    
    参数:
    -----------
    tickers : list
        要扫描的股票代码列表
    ref_ticker : str
        用于相对强弱比较的参考股票代码
    timeframes : list, optional
        要分析的多个时间周期列表，例如 [("5m", "1d"), ("1h", "5d")]
    warning_threshold : int
        发出警告的信号强度阈值 (0-14)，默认为11分（约80%）
    debug_no_signal : bool
        是否调试无信号的股票
        
    返回:
    --------
    list : 满足警告阈值的信号列表
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{now}] 扫描美国十大巨头股票的趋势反转信号...")
    
    # 如果没有指定时间周期，使用默认的单一时间周期
    if timeframes is None:
        timeframes = [("1m", "1d")]  # 默认使用1分钟K线，1天数据
        
    print(f"分析的时间周期: {', '.join([f'{interval}({period})' for interval, period in timeframes])}")
    
    signals = []
    warning_signals = []
    no_signal_stocks = []
    
    # 创建结果表格的标题
    print("\n" + "="*90)
    print(f"{'股票代码':<8} {'时间周期':<8} {'信号类型':<6} {'信号强度':<8} {'信号等级':<10} {'背离':<6} {'成交量':<6} {'相对强弱':<8} {'支撑/阻力':<8}")
    print("-"*90)
    
    for ticker in tickers:
        try:
            # 分析股票但不打印详细信息
            result = analyze_stock(ticker, ref_ticker, timeframes=timeframes, verbose=False)
            
            if result is None or result['signal_type'] is None:
                # 打印无信号的结果
                print(f"{ticker:<8} {'所有':<8} {'无信号':<6} {'-':<8} {'-':<10} {'-':<6} {'-':<6} {'-':<8} {'-':<8}")
                no_signal_stocks.append(ticker)
                continue
                
            # 添加到信号列表
            signals.append(result)
            
            # 打印结果行
            timeframe = result.get('timeframe', '未知')
            print(f"{result['ticker']:<8} {timeframe:<8} {result['signal_type']:<6} {result['signal_strength']:<8} {result['signal_class']:<10} {result['details']['divergence_score']:<6} {result['details']['volume_score']:<6} {result['details']['relative_strength_score']:<6} {result['details']['support_resistance_score']:<6}")
            
            # 检查是否超过警告阈值
            if result['signal_strength'] >= warning_threshold:
                warning_signals.append(result)
                
        except Exception as e:
            print(f"分析 {ticker} 时出错: {str(e)}")
            print(traceback.format_exc())
    
    print("="*90)
    
    # 打印警告信息
    if warning_signals:
        print(f"\n⚠️  警告! 发现 {len(warning_signals)} 个强烈的反转信号 ⚠️")
        for signal in warning_signals:
            try:
                direction = "看跌" if signal['signal_type'] == "PUT" else "看涨"
                timeframe = signal.get('timeframe', '未知')
                print(f"{signal['ticker']} 在 {timeframe} 时间周期上出现强烈的{direction}信号! 信号强度: {signal['signal_strength']}/14, 成功概率: {signal['success_probability']}")
                
                # 为警告信号绘制图表
                if 'data_period' in signal and 'timeframe' in signal:
                    analyze_single_timeframe(signal['ticker'], ref_ticker, signal['data_period'], signal['timeframe'], verbose=True)
                else:
                    analyze_single_timeframe(signal['ticker'], ref_ticker, "1d", "1m", verbose=True)
                
            except Exception as e:
                print(f"处理警告信号时出错 {signal['ticker']}: {str(e)}")
                print(traceback.format_exc())
    
    # 如果需要调试无信号的股票
    if debug_no_signal and no_signal_stocks:
        print(f"\n发现 {len(no_signal_stocks)} 个无信号股票: {', '.join(no_signal_stocks)}")
        debug_no_signal_stocks(no_signal_stocks, ref_ticker, timeframes, sensitivity)
    
    return warning_signals

def run_scheduled_scan(timeframes=None, sensitivity=1.0):
    """
    执行定时扫描任务
    
    参数:
    -----------
    timeframes : list, optional
        要分析的多个时间周期列表，例如 [("5m", "1d"), ("1h", "5d")]
    sensitivity : float
        背离检测的敏感度，值越大越容易检测到背离
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            print(f"尝试执行扫描 (尝试 {retry_count + 1}/{max_retries})...")
            # 应用补丁
            patch_trend_reversal_signal(sensitivity)
            # 执行扫描
            scan_top_stocks(US_TOP_10, timeframes=timeframes, warning_threshold=11, debug_no_signal=False)
            # 如果成功执行，跳出循环
            break
        except Exception as e:
            retry_count += 1
            print(f"执行扫描时出错: {str(e)}")
            print(traceback.format_exc())
            if retry_count < max_retries:
                print(f"将在5秒后重试...")
                time.sleep(5)
            else:
                print("达到最大重试次数，放弃扫描")

def backtest_stock(ticker: str, args: TrendArgs, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
    """
    对单个股票进行回测。
    
    参数:
    -----------
    ticker : str
        要回测的股票代码
    args : TrendArgs
        趋势反转策略参数配置
    start_date : datetime, optional
        回测开始日期，如果不指定则使用args中的配置
    end_date : datetime, optional
        回测结束日期，如果不指定则使用args中的配置
        
    返回:
    --------
    List[Dict] : 回测结果列表，每个元素包含一个信号的详细信息
    """
    start_date = start_date or args.backtest_start
    end_date = end_date or args.backtest_end
    
    if args.verbose:
        print(f"\n开始回测 {ticker} 从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    
    results = []
    
    # 对每个时间周期进行回测
    for interval, period in args.timeframes:
        try:
            # 根据不同的时间周期调整数据下载范围
            actual_start = start_date
            if interval == "1m":
                # 1分钟数据只能获取最近7天
                one_week_ago = end_date - timedelta(days=7)
                if start_date < one_week_ago:
                    if args.verbose:
                        print(f"警告: 1分钟数据只能获取最近一周的数据，将开始日期从 {start_date.strftime('%Y-%m-%d')} 调整为 {one_week_ago.strftime('%Y-%m-%d')}")
                    actual_start = one_week_ago
            elif interval == "5m":
                # 5分钟数据只能获取最近60天
                sixty_days_ago = end_date - timedelta(days=60)
                if start_date < sixty_days_ago:
                    if args.verbose:
                        print(f"警告: 5分钟数据只能获取最近60天的数据，将开始日期从 {start_date.strftime('%Y-%m-%d')} 调整为 {sixty_days_ago.strftime('%Y-%m-%d')}")
                    actual_start = sixty_days_ago
            elif interval == "15m":
                # 15分钟数据只能获取最近1天
                yesterday = end_date - timedelta(days=1)
                if start_date < yesterday:
                    if args.verbose:
                        print(f"警告: 15分钟数据只能获取最近1天的数据，将开始日期从 {start_date.strftime('%Y-%m-%d')} 调整为 {yesterday.strftime('%Y-%m-%d')}")
                    actual_start = yesterday
            
            # 下载数据
            stock_data = yf.download(ticker, start=actual_start, end=end_date, interval=interval)
            ref_data = yf.download(args.reference_ticker, start=actual_start, end=end_date, interval=interval)
            
            if stock_data.empty or ref_data.empty:
                if args.verbose:
                    print(f"错误: 无法下载 {ticker} 或 {args.reference_ticker} 的数据")
                continue
            
            if args.verbose:
                print(f"\n分析 {interval} 时间周期的数据...")
                print(f"下载了 {len(stock_data)} 条 {ticker} 数据记录")
                print(f"下载了 {len(ref_data)} 条 {args.reference_ticker} 数据记录")
            
            # 统一处理DataFrame
            stock_data = process_dataframe(stock_data, ticker)
            ref_data = process_dataframe(ref_data, args.reference_ticker)
            
            # 初始化信号计算器
            signal = TrendReversalSignal(ticker, args.reference_ticker, args)
            signal.load_data(stock_data, ref_data)
            
            # 计算信号强度
            result = signal.calculate_signal_strength()
            
            if result and result['signal_type'] is not None:
                # 添加时间戳和时间周期信息
                result['timestamp'] = stock_data.index[-1]
                result['timeframe'] = interval
                result['data_period'] = period
                
                # 如果信号强度超过最小阈值，添加到结果列表
                if result['signal_strength'] >= args.min_signal_strength:
                    results.append(result)
                    
                    if args.verbose:
                        print(f"\n=== {ticker} ({interval}) 趋势反转信号分析 ===")
                        print(f"时间: {result['timestamp']}")
                        print(f"信号类型: {result['signal_type']}")
                        print(f"信号强度: {result['signal_strength']}/14")
                        print(f"信号等级: {result['signal_class']}")
                        print(f"成功概率: {result['success_probability']}")
                        print(f"建议仓位大小: {result['recommended_position']}")
                        print("\n详细得分:")
                        print(f"  背离得分: {result['details']['divergence_score']}/4")
                        print(f"  成交量得分: {result['details']['volume_score']}/4")
                        print(f"  相对强弱得分: {result['details']['relative_strength_score']}/3")
                        print(f"  支撑/阻力得分: {result['details']['support_resistance_score']}/3")
                    
                    # 如果需要绘制图表
                    if args.plot_charts:
                        signal.plot_signal()
        
        except Exception as e:
            if args.verbose:
                print(f"分析 {ticker} ({interval}) 时出错: {str(e)}")
                if args.debug_mode:
                    print(traceback.format_exc())
    
    return results

def backtest_portfolio(args: TrendArgs) -> Dict:
    """
    对整个投资组合进行回测。
    
    参数:
    -----------
    args : TrendArgs
        趋势反转策略参数配置
        
    返回:
    --------
    Dict : 回测结果统计
    """
    all_results = []
    stats = {
        "total_signals": 0,
        "call_signals": 0,
        "put_signals": 0,
        "strong_signals": 0,
        "extreme_signals": 0,
        "signals_by_stock": {},
        "signals_by_timeframe": {},
        "average_strength": 0.0
    }
    
    print("\n=== 开始投资组合回测 ===")
    print(f"回测周期: {args.backtest_start.strftime('%Y-%m-%d')} 到 {args.backtest_end.strftime('%Y-%m-%d')}")
    print(f"分析的时间周期: {', '.join([f'{interval}({period})' for interval, period in args.timeframes])}")
    print(f"股票列表: {', '.join(args.tickers)}")
    print("="*50)
    
    for ticker in args.tickers:
        # 回测单个股票
        results = backtest_stock(ticker, args)
        all_results.extend(results)
        
        # 更新统计信息
        stats["signals_by_stock"][ticker] = len(results)
        
        for result in results:
            stats["total_signals"] += 1
            if result["signal_type"] == "CALL":
                stats["call_signals"] += 1
            else:
                stats["put_signals"] += 1
            
            if result["signal_strength"] >= args.warning_threshold:
                stats["extreme_signals"] += 1
            elif result["signal_strength"] >= args.min_signal_strength:
                stats["strong_signals"] += 1
            
            timeframe = result["timeframe"]
            if timeframe not in stats["signals_by_timeframe"]:
                stats["signals_by_timeframe"][timeframe] = 0
            stats["signals_by_timeframe"][timeframe] += 1
    
    # 计算平均信号强度
    if all_results:
        stats["average_strength"] = sum(r["signal_strength"] for r in all_results) / len(all_results)
    
    # 打印回测统计
    print("\n=== 回测统计 ===")
    print(f"总信号数: {stats['total_signals']}")
    print(f"看涨信号: {stats['call_signals']}")
    print(f"看跌信号: {stats['put_signals']}")
    print(f"强信号数: {stats['strong_signals']}")
    print(f"极强信号数: {stats['extreme_signals']}")
    print(f"平均信号强度: {stats['average_strength']:.2f}")
    
    print("\n按股票统计:")
    for ticker, count in stats["signals_by_stock"].items():
        print(f"{ticker}: {count}个信号")
    
    print("\n按时间周期统计:")
    for timeframe, count in stats["signals_by_timeframe"].items():
        print(f"{timeframe}: {count}个信号")
    
    # 如果需要保存结果
    if args.save_results:
        save_backtest_results(all_results, stats, args)
    
    return stats

def save_backtest_results(results: List[Dict], stats: Dict, args: TrendArgs) -> None:
    """
    保存回测结果到文件。
    
    参数:
    -----------
    results : List[Dict]
        所有信号的详细信息
    stats : Dict
        回测统计信息
    args : TrendArgs
        策略参数配置
    """
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"backtest_{timestamp}"
    
    # 保存详细结果
    results_file = os.path.join(results_dir, f"{base_filename}_signals.json")
    with open(results_file, 'w') as f:
        # 将datetime对象转换为字符串
        serializable_results = []
        for r in results:
            r_copy = r.copy()
            if 'timestamp' in r_copy:
                r_copy['timestamp'] = r_copy['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=4)
    
    # 保存统计信息
    stats_file = os.path.join(results_dir, f"{base_filename}_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    # 保存参数配置
    config_file = os.path.join(results_dir, f"{base_filename}_config.json")
    args.save_to_file(config_file)
    
    print(f"\n回测结果已保存到 {results_dir} 目录")

def run_backtest(config_file: str = None) -> None:
    """
    运行回测。
    
    参数:
    -----------
    config_file : str, optional
        参数配置文件路径
    """
    # 加载参数配置
    if config_file and os.path.exists(config_file):
        args = TrendArgs.load_from_file(config_file)
        print(f"从 {config_file} 加载参数配置")
    else:
        args = TrendArgs()
        print("使用默认参数配置")
    
    # 设置回测模式
    args.debug_mode = True
    
    # 运行回测
    stats = backtest_portfolio(args)
    
    return stats

def main():
    """主函数 - 根据命令行参数执行不同的功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description='趋势反转策略实时监控和回测')
    parser.add_argument('--mode', type=str, choices=['monitor', 'backtest'], default='monitor',
                      help='运行模式: monitor(实时监控) 或 backtest(回测)')
    parser.add_argument('--config', type=str, help='参数配置文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--stocks', type=str, help='要分析的股票代码，用逗号分隔')
    parser.add_argument('--timeframes', type=str, help='要分析的时间周期，格式为interval1:period1,interval2:period2')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='背离检测的敏感度，值越大越容易检测到背离')
    
    args = parser.parse_args()
    
    if args.mode == 'backtest':
        # 运行回测
        run_backtest(args.config)
    else:
        # 运行实时监控
        # 加载参数配置
        if args.config and os.path.exists(args.config):
            trend_args = TrendArgs.load_from_file(args.config)
            print(f"从 {args.config} 加载参数配置")
        else:
            trend_args = TrendArgs()
            print("使用默认参数配置")
        
        # 更新参数
        if args.debug:
            trend_args.debug_mode = True
        if args.stocks:
            trend_args.tickers = args.stocks.split(',')
        if args.timeframes:
            trend_args.timeframes = []
            for tf in args.timeframes.split(','):
                interval, period = tf.split(':')
                trend_args.timeframes.append((interval, period))
        trend_args.divergence_sensitivity = args.sensitivity
        
        print("\n趋势反转策略实时监控")
        print("-------------------------------")
        print("此程序将定期扫描股票的趋势反转信号")
        print(f"当信号强度超过{trend_args.warning_threshold}分时将发出警告")
        print("按Ctrl+C停止程序")
        print("-------------------------------")
        
        # 执行初始扫描
        run_scheduled_scan(trend_args.timeframes, trend_args.divergence_sensitivity)
        
        # 设置定时任务 - 每5分钟执行一次
        schedule.every(5).minutes.do(lambda: run_scheduled_scan(trend_args.timeframes, trend_args.divergence_sensitivity))
        
        # 运行定时任务循环
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n程序已停止")

if __name__ == "__main__":
    main()

    # 运行实时监控:
    # python main.py --mode monitor --config config.json
    
    # 运行回测:
    # python main.py --mode backtest --config config.json
    
    # 调试模式:
    # python main.py --mode monitor --debug --timeframes 5m:1d,1h:5d --sensitivity 2.0