#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
趋势反转信号计算器
--------------------------------
此脚本计算短期趋势反转的信号强度，
基于技术指标、成交量分析、相对强弱和支撑/阻力位分析。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
from datetime import datetime, timedelta
import traceback

class TrendReversalSignal:
    def __init__(self, ticker, reference_ticker="QQQ", args=None):
        """
        初始化趋势反转信号计算器。
        
        参数:
        -----------
        ticker : str
            要分析的股票代码
        reference_ticker : str
            用于相对强弱比较的参考股票代码（默认: QQQ）
        args : TrendArgs, optional
            趋势反转策略参数配置
        """
        self.ticker = ticker
        self.reference_ticker = reference_ticker
        self.args = args
        self.signal_strength = 0
        self.signal_details = {
            "divergence": 0,
            "volume": 0,
            "relative_strength": 0,
            "support_resistance": 0
        }
        self.signal_type = None  # "CALL" 或 "PUT"
        
    def load_data(self, data, ref_data=None):
        """
        加载价格和成交量数据进行分析。
        
        参数:
        -----------
        data : pandas.DataFrame
            包含目标股票OHLCV数据的DataFrame
        ref_data : pandas.DataFrame
            包含参考股票OHLCV数据的DataFrame
        """
        self.data = data
        self.ref_data = ref_data
        
        # 计算基本技术指标
        self._calculate_indicators()
        
    def _calculate_indicators(self):
        """计算用于背离分析的技术指标"""
        # 使用配置参数计算指标
        rsi_length = self.args.rsi_length if self.args else 14
        macd_fast = self.args.macd_fast if self.args else 12
        macd_slow = self.args.macd_slow if self.args else 26
        macd_signal = self.args.macd_signal if self.args else 9
        stoch_k = self.args.stoch_k if self.args else 5
        stoch_d = self.args.stoch_d if self.args else 3
        stoch_smooth = self.args.stoch_smooth if self.args else 3
        sma_periods = self.args.sma_periods if self.args else [20, 50, 200]
        
        # RSI
        self.data['rsi'] = ta.rsi(self.data['close'], length=rsi_length)
        
        # MACD
        macd = ta.macd(self.data['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        self.data['macd'] = macd['MACD_{}_{}_{}'.format(macd_fast, macd_slow, macd_signal)]
        self.data['macd_signal'] = macd['MACDs_{}_{}_{}'.format(macd_fast, macd_slow, macd_signal)]
        self.data['macd_hist'] = macd['MACDh_{}_{}_{}'.format(macd_fast, macd_slow, macd_signal)]
        
        # 随机指标
        stoch = ta.stoch(self.data['high'], self.data['low'], self.data['close'], 
                        k=stoch_k, d=stoch_d, smooth_k=stoch_smooth)
        self.data['slowk'] = stoch['STOCHk_{}_{}_{}'.format(stoch_k, stoch_d, stoch_smooth)]
        self.data['slowd'] = stoch['STOCHd_{}_{}_{}'.format(stoch_k, stoch_d, stoch_smooth)]
        
        # 移动平均线
        for period in sma_periods:
            self.data[f'sma{period}'] = ta.sma(self.data['close'], length=period)
        
        # 成交量指标
        self.data['volume_sma20'] = ta.sma(self.data['volume'], length=20)
        
    def analyze_divergence(self, lookback=None):
        """
        分析价格-指标背离。
        
        参数:
        -----------
        lookback : int, optional
            用于背离分析的回溯周期数
            
        返回:
        --------
        int : 背离得分 (0-4)
        """
        # 使用配置参数
        if self.args:
            lookback = lookback or self.args.divergence_lookback
            sensitivity = self.args.divergence_sensitivity
            min_price_change = self.args.min_price_change
            min_indicator_change = self.args.min_indicator_change
        else:
            lookback = lookback or 20
            sensitivity = 1.0
            min_price_change = 0.0
            min_indicator_change = 0.0
        
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
            
            # 检查价格是否创新高 - 使用敏感度
            if price_change > -1.0 * sensitivity:  # 允许轻微下降
                bearish_divs = []
                for indicator in ['rsi', 'macd', 'slowk']:
                    ind_change = (recent_data.loc[last_two_highs[-1], indicator] / 
                                 recent_data.loc[last_two_highs[-2], indicator] - 1) * 100
                    # 如果指标没有创新高，则为顶背离 - 使用敏感度
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
                        # 使用敏感度
                        if not (ind_vals[0] > ind_vals[1] * (1 - 0.05/sensitivity) and 
                                ind_vals[1] > ind_vals[2] * (1 - 0.05/sensitivity)):
                            triple_div = False
                    
                    if triple_div and len(bearish_divs) > 0:
                        divergence_score = 3  # 确保三重背离获得高分
                
                # 检查背离幅度 - 使用敏感度
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
            
            # 检查价格是否创新低 - 使用敏感度
            if price_change < 1.0 * sensitivity:  # 允许轻微上升
                bullish_divs = []
                for indicator in ['rsi', 'macd', 'slowk']:
                    ind_change = (recent_data.loc[last_two_lows[-1], indicator] / 
                                 recent_data.loc[last_two_lows[-2], indicator] - 1) * 100
                    # 如果指标没有创新低，则为底背离 - 使用敏感度
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
                        # 使用敏感度
                        if not (ind_vals[0] < ind_vals[1] * (1 + 0.05/sensitivity) and 
                                ind_vals[1] < ind_vals[2] * (1 + 0.05/sensitivity)):
                            triple_div = False
                    
                    if triple_div and len(bullish_divs) > 0:
                        divergence_score = 3  # 确保三重背离获得高分
                
                # 检查背离幅度 - 使用敏感度
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
    
    def analyze_volume(self, pivot_bars=None, lookback=None):
        """
        分析潜在反转点周围的成交量模式。
        
        参数:
        -----------
        pivot_bars : int, optional
            分析枢轴点周围的K线数量
        lookback : int, optional
            用于成交量分析的回溯周期数
            
        返回:
        --------
        int : 成交量得分 (0-4)
        """
        if self.signal_type is None:
            return 0
            
        # 使用配置参数
        if self.args:
            pivot_bars = pivot_bars or self.args.volume_pivot_bars
            lookback = lookback or self.args.volume_lookback
            min_volume_ratio = self.args.min_volume_ratio
            high_volume_threshold = self.args.high_volume_threshold
        else:
            pivot_bars = pivot_bars or 5
            lookback = lookback or 30
            min_volume_ratio = 3.0
            high_volume_threshold = 2.0
        
        recent_data = self.data.tail(lookback).copy()
        
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
        
        # 获取枢轴点周围的成交量
        pivot_start = max(0, pivot_pos - pivot_bars)
        pivot_end = min(len(recent_data) - 1, pivot_pos + pivot_bars)
        
        pivot_volume = recent_data.iloc[pivot_start:pivot_end+1]['volume'].mean()
        
        # 获取基准成交量（枢轴点之前）
        baseline_start = max(0, pivot_start - 20)
        baseline_end = pivot_start - 1
        
        if baseline_end < baseline_start:
            return 0
        
        baseline_volume = recent_data.iloc[baseline_start:baseline_end+1]['volume'].mean()
        
        # 计算成交量比率
        if baseline_volume == 0:
            return 0
        
        volume_ratio = pivot_volume / baseline_volume
        
        # 根据成交量比率评分
        volume_score = 0
        if volume_ratio >= min_volume_ratio and volume_ratio < min_volume_ratio + 1:
            volume_score = 1
        elif volume_ratio >= min_volume_ratio + 1 and volume_ratio < min_volume_ratio + 2:
            volume_score = 2
        elif volume_ratio >= min_volume_ratio + 2:
            volume_score = 3
        
        # 检查成交量放大是否持续多根K线
        high_vol_bars = sum(recent_data.iloc[pivot_start:pivot_end+1]['volume'] > 
                           high_volume_threshold * baseline_volume)
        
        if high_vol_bars >= 3:
            volume_score += 1
        
        # 将分数上限设为4
        volume_score = min(volume_score, 4)
        
        self.signal_details["volume"] = volume_score
        self.volume_ratio = volume_ratio
        
        return volume_score
    
    def analyze_relative_strength(self, lookback=None):
        """
        分析相对强弱。
        
        参数:
        -----------
        lookback : int, optional
            用于相对强弱分析的回溯天数
            
        返回:
        --------
        int : 相对强弱得分 (0-3)
        """
        try:
            if self.signal_type is None or self.ref_data is None:
                return 0
            
            # 使用参数或默认值
            lookback = lookback or getattr(self.args, 'rs_lookback', 1)
            threshold_bullish = getattr(self.args, 'rs_threshold_bullish', 1.05)
            threshold_bearish = getattr(self.args, 'rs_threshold_bearish', 0.95)
            
            # 确保数据足够
            if len(self.data) <= lookback or len(self.ref_data) <= lookback:
                return 0
            
            # 计算相对强弱
            ticker_perf = (self.data['close'].iloc[-1] / self.data['close'].iloc[-lookback-1] - 1)
            ref_perf = (self.ref_data['close'].iloc[-1] / self.ref_data['close'].iloc[-lookback-1] - 1)
            
            if ref_perf == 0:  # 避免除以零
                relative_strength = 1.0
            else:
                relative_strength = (1 + ticker_perf) / (1 + ref_perf)
            
            # 根据相对强弱和信号类型评分
            rs_score = 0
            
            if self.signal_type == "CALL":
                # 看涨信号 - 相对强弱应该较弱（超卖）
                if relative_strength < threshold_bearish:
                    rs_score = 2
                elif relative_strength < 1.0:
                    rs_score = 1
                
                # 检查是否有反转迹象
                if lookback >= 3:
                    short_rs = (self.data['close'].iloc[-1] / self.data['close'].iloc[-3] - 1) / \
                              (self.ref_data['close'].iloc[-1] / self.ref_data['close'].iloc[-3] - 1)
                    if short_rs > 1.0 and relative_strength < 1.0:
                        rs_score += 1
            
            elif self.signal_type == "PUT":
                # 看跌信号 - 相对强弱应该较强（超买）
                if relative_strength > threshold_bullish:
                    rs_score = 2
                elif relative_strength > 1.0:
                    rs_score = 1
                
                # 检查是否有反转迹象
                if lookback >= 3:
                    short_rs = (self.data['close'].iloc[-1] / self.data['close'].iloc[-3] - 1) / \
                              (self.ref_data['close'].iloc[-1] / self.ref_data['close'].iloc[-3] - 1)
                    if short_rs < 1.0 and relative_strength > 1.0:
                        rs_score += 1
            
            # 将分数上限设为3
            rs_score = min(rs_score, 3)
            
            self.signal_details["relative_strength"] = rs_score
            self.relative_strength = relative_strength
            
            return rs_score
        except Exception as e:
            print(f"分析相对强弱时出错: {str(e)}")
            return 0
    
    def analyze_support_resistance(self, lookback=None):
        """
        分析价格是否接近重要支撑/阻力位。
        
        参数:
        -----------
        lookback : int, optional
            用于支撑/阻力分析的回溯周期数
            
        返回:
        --------
        int : 支撑/阻力得分 (0-3)
        """
        try:
            if self.signal_type is None:
                return 0
            
            # 使用参数或默认值
            price_threshold = getattr(self.args, 'sr_price_threshold', 0.05)
            cluster_threshold = getattr(self.args, 'sr_cluster_threshold', 0.01)
            lookback = lookback or getattr(self.args, 'sr_lookback', 100)
            
            # 确保数据足够
            if len(self.data) < lookback:
                lookback = len(self.data) - 1
            
            if lookback < 10:
                return 0
            
            hist_data = self.data.tail(lookback).copy()
            current_price = self.data['close'].iloc[-1]  # 使用 iloc 安全地获取最后一个价格
            
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
            hist_data['price_high'] = hist_data['high'].rolling(10, center=True).apply(
                lambda x: x[5] if len(x) > 5 and x[5] == max(x) else np.nan, raw=True)
            hist_data['price_low'] = hist_data['low'].rolling(10, center=True).apply(
                lambda x: x[5] if len(x) > 5 and x[5] == min(x) else np.nan, raw=True)
            
            highs = hist_data['price_high'].dropna().tolist()
            lows = hist_data['price_low'].dropna().tolist()
            
            sr_levels.extend(highs + lows)
            
            # 3. 移动平均线
            try:
                latest = self.data.iloc[-1]  # 使用 iloc 安全地获取最后一行
                ma_levels = []
                for ma in ['sma20', 'sma50', 'sma200']:
                    if ma in latest and not pd.isna(latest[ma]):
                        ma_levels.append(latest[ma])
                sr_levels.extend(ma_levels)
            except Exception as e:
                print(f"获取移动平均线时出错: {str(e)}")
            
            # 寻找距离当前价格最近的水平
            sr_levels = [level for level in sr_levels if not np.isnan(level)]
            sr_levels.sort()
            
            # 寻找在当前价格price_threshold范围内的水平
            nearby_levels = [level for level in sr_levels 
                            if abs(level/current_price - 1) < price_threshold]
            
            # 计算重叠水平（彼此相差cluster_threshold以内）
            clusters = []
            for level in nearby_levels:
                added = False
                for cluster in clusters:
                    if any(abs(level/l - 1) < cluster_threshold for l in cluster):
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
    
    def calculate_signal_strength(self):
        """
        基于所有因素计算整体信号强度。
        
        返回:
        --------
        dict : 包含强度分数和分类的信号详情
        """
        # 重置信号强度
        self.signal_strength = 0
        
        # 分析每个组成部分
        div_score = self.analyze_divergence()
        vol_score = self.analyze_volume()
        rs_score = self.analyze_relative_strength()
        sr_score = self.analyze_support_resistance()
        
        # 计算总分
        total_score = div_score + vol_score + rs_score + sr_score
        
        # 确定信号分类
        if self.args:
            min_signal_strength = self.args.min_signal_strength
            warning_threshold = self.args.warning_threshold
        else:
            min_signal_strength = 8
            warning_threshold = 11
        
        if total_score <= min_signal_strength - 4:
            signal_class = "弱信号"
            probability = "<40%"
            position_size = "0-10%"
        elif total_score <= min_signal_strength:
            signal_class = "中等信号"
            probability = "40-60%"
            position_size = "10-30%"
        elif total_score <= warning_threshold:
            signal_class = "强信号"
            probability = "60-75%"
            position_size = "30-50%"
        else:
            signal_class = "极强信号"
            probability = ">75%"
            position_size = "50-70%"
        
        self.signal_strength = total_score
        
        result = {
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "signal_strength": total_score,
            "signal_class": signal_class,
            "success_probability": probability,
            "recommended_position": position_size,
            "details": {
                "divergence_score": div_score,
                "volume_score": vol_score,
                "relative_strength_score": rs_score,
                "support_resistance_score": sr_score
            }
        }
        
        return result
    
    def plot_signal(self, lookback=30):
        """
        绘制信号图表。
        
        参数:
        -----------
        lookback : int
            要显示的回溯周期数
        """
        try:
            if self.signal_type is None:
                print("没有检测到信号，无法绘制图表")
                return
            
            # 获取最近数据
            recent_data = self.data.tail(lookback).copy()
            
            # 创建图表
            fig, axs = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
            fig.suptitle(f"{self.ticker} 趋势反转信号分析 ({self.signal_type})", fontsize=16)
            
            # 绘制价格和移动平均线
            axs[0].set_title("价格和移动平均线")
            axs[0].plot(recent_data.index, recent_data['close'], label='收盘价')
            
            for ma in ['sma20', 'sma50', 'sma200']:
                if ma in recent_data.columns:
                    axs[0].plot(recent_data.index, recent_data[ma], label=ma.upper())
            
            # 标记信号
            if self.signal_type == "CALL":
                axs[0].scatter(recent_data.index[-1], recent_data['close'].iloc[-1], 
                              color='green', s=100, marker='^', label='看涨信号')
            else:
                axs[0].scatter(recent_data.index[-1], recent_data['close'].iloc[-1], 
                              color='red', s=100, marker='v', label='看跌信号')
            
            # 添加支撑/阻力位
            if hasattr(self, 'sr_levels') and self.sr_levels:
                for level in self.sr_levels:
                    axs[0].axhline(y=level, color='purple', linestyle='--', alpha=0.5)
            
            axs[0].set_ylabel("价格")
            axs[0].legend()
            axs[0].grid(True)
            
            # 绘制成交量
            axs[1].set_title("成交量")
            axs[1].bar(recent_data.index, recent_data['volume'], color='blue', alpha=0.5)
            axs[1].set_ylabel("成交量")
            axs[1].grid(True)
            
            # 绘制技术指标
            axs[2].set_title("技术指标")
            
            # 绘制RSI
            if 'rsi' in recent_data.columns:
                axs[2].plot(recent_data.index, recent_data['rsi'], label='RSI', color='orange')
                axs[2].axhline(y=70, color='red', linestyle='--', alpha=0.5)
                axs[2].axhline(y=30, color='green', linestyle='--', alpha=0.5)
            
            # 绘制MACD
            if 'macd' in recent_data.columns and 'macd_signal' in recent_data.columns:
                axs[2].plot(recent_data.index, recent_data['macd'], label='MACD', color='blue')
                axs[2].plot(recent_data.index, recent_data['macd_signal'], label='信号线', color='red')
            
            # 绘制随机指标
            if 'slowk' in recent_data.columns and 'slowd' in recent_data.columns:
                axs[2].plot(recent_data.index, recent_data['slowk'], label='%K', color='purple')
                axs[2].plot(recent_data.index, recent_data['slowd'], label='%D', color='magenta')
                axs[2].axhline(y=80, color='red', linestyle='--', alpha=0.5)
                axs[2].axhline(y=20, color='green', linestyle='--', alpha=0.5)
            
            axs[2].set_ylabel("指标值")
            axs[2].legend()
            axs[2].grid(True)
            
            # 添加信号强度信息
            strength_text = f"信号强度: {sum(self.signal_details.values())}/14\n"
            for key, value in self.signal_details.items():
                strength_text += f"{key}: {value}\n"
            
            plt.figtext(0.01, 0.01, strength_text, fontsize=10)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"绘制图表时出错: {str(e)}")
            print(traceback.format_exc())


def demo_with_sample_data():
    """生成样本数据并演示信号计算器"""
    # 生成样本价格数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
    
    # 创建带有顶背离模式的样本股票数据
    close = np.linspace(100, 150, 80)  # 上升趋势
    close = np.append(close, np.linspace(150, 140, 20))  # 末尾轻微下降趋势
    
    # 添加一些噪声
    close = close + np.random.normal(0, 2, 100)
    
    # 创建末尾有放量的成交量
    volume = np.random.normal(1000, 200, 100)
    volume[-10:] = volume[-10:] * 3  # 末尾成交量放大
    
    # 创建样本数据
    stock_data = pd.DataFrame({
        'open': close - np.random.normal(0, 1, 100),
        'high': close + np.random.normal(2, 1, 100),
        'low': close - np.random.normal(2, 1, 100),
        'close': close,
        'volume': volume
    }, index=dates)
    
    # 创建参考指数数据（如QQQ）
    ref_close = np.linspace(400, 430, 80)  # 上升趋势
    ref_close = np.append(ref_close, np.linspace(430, 425, 20))  # 轻微下降
    
    ref_data = pd.DataFrame({
        'open': ref_close - np.random.normal(0, 1, 100),
        'high': ref_close + np.random.normal(2, 1, 100),
        'low': ref_close - np.random.normal(2, 1, 100),
        'close': ref_close,
        'volume': np.random.normal(5000, 1000, 100)
    }, index=dates)
    
    # 初始化信号计算器
    signal = TrendReversalSignal("样本", "QQQ")
    signal.load_data(stock_data, ref_data)
    
    # 计算信号强度
    result = signal.calculate_signal_strength()
    
    # 打印结果
    print("\n=== 趋势反转信号分析 ===")
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
    signal.plot_signal()


if __name__ == "__main__":
    print("趋势反转信号计算器")
    print("--------------------------------")
    print("此工具计算短期趋势反转的信号强度。")
    
    # 运行样本数据演示
    demo_with_sample_data() 