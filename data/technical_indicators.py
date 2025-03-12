import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import List, Optional, Union, Dict
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log.logger import init_logger

logger = init_logger('technical_indicators.log')


class TechnicalIndicatorProcessor:
    """
    技术指标处理器 - 计算和添加各种技术分析指标
    """
    
    def __init__(self):
        """初始化技术指标处理器"""
        pass
    
    def add_moving_averages(self, 
                           df: pd.DataFrame,
                           periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        添加简单移动平均线(SMA)
        
        参数:
            df: 包含价格数据的DataFrame
            periods: 移动平均线周期列表
            
        返回:
            添加了移动平均线的DataFrame
        """
        result_df = df.copy()
        
        for period in periods:
            col_name = f'ma_{period}'
            result_df[col_name] = ta.sma(result_df['close'], length=period)
            
        return result_df
    
    def add_exponential_moving_averages(self, 
                                       df: pd.DataFrame,
                                       periods: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        添加指数移动平均线(EMA)
        
        参数:
            df: 包含价格数据的DataFrame
            periods: 移动平均线周期列表
            
        返回:
            添加了指数移动平均线的DataFrame
        """
        result_df = df.copy()
        
        for period in periods:
            col_name = f'ema_{period}'
            result_df[col_name] = ta.ema(result_df['close'], length=period)
            
        return result_df
    
    def add_bollinger_bands(self, 
                           df: pd.DataFrame,
                           period: int = 20,
                           std_dev: float = 2.0) -> pd.DataFrame:
        """
        添加布林带指标
        
        参数:
            df: 包含价格数据的DataFrame
            period: 移动平均线周期
            std_dev: 标准差乘数
            
        返回:
            添加了布林带的DataFrame
        """
        result_df = df.copy()

        if df.__len__() <= 20:
            return result_df
        
        bb = ta.bbands(result_df['close'], length=period, std=std_dev)
        
        result_df['bb_upper'] = bb['BBU_' + str(period) + '_' + str(std_dev)]
        result_df['bb_middle'] = bb['BBM_' + str(period) + '_' + str(std_dev)]
        result_df['bb_lower'] = bb['BBL_' + str(period) + '_' + str(std_dev)]
        result_df['bb_width'] = bb['BBB_' + str(period) + '_' + str(std_dev)]
        
        return result_df
    
    def add_rsi(self, 
               df: pd.DataFrame,
               periods: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """
        添加相对强弱指数(RSI)
        
        参数:
            df: 包含价格数据的DataFrame
            periods: RSI周期列表
            
        返回:
            添加了RSI的DataFrame
        """
        result_df = df.copy()
        
        for period in periods:
            col_name = f'rsi_{period}'
            result_df[col_name] = ta.rsi(result_df['close'], length=period)
            
        return result_df
    
    def add_macd(self, 
                df: pd.DataFrame,
                fast: int = 12,
                slow: int = 26,
                signal: int = 9) -> pd.DataFrame:
        """
        添加MACD指标
        
        参数:
            df: 包含价格数据的DataFrame
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        返回:
            添加了MACD的DataFrame
        """
        result_df = df.copy()
        
        macd = ta.macd(result_df['close'], fast=fast, slow=slow, signal=signal)
        
        result_df['macd'] = macd[f'MACD_{fast}_{slow}_{signal}']
        result_df['macd_signal'] = macd[f'MACDs_{fast}_{slow}_{signal}']
        result_df['macd_hist'] = macd[f'MACDh_{fast}_{slow}_{signal}']
        
        return result_df
    
    def add_stochastic_oscillator(self, 
                                 df: pd.DataFrame,
                                 k_period: int = 14,
                                 d_period: int = 3,
                                 smooth_k: int = 3) -> pd.DataFrame:
        """
        添加随机振荡器指标
        
        参数:
            df: 包含价格数据的DataFrame
            k_period: %K周期
            d_period: %D周期
            smooth_k: %K平滑参数
            
        返回:
            添加了随机振荡器的DataFrame
        """
        result_df = df.copy()
        
        stoch = ta.stoch(high=result_df['high'], low=result_df['low'], close=result_df['close'], 
                         k=k_period, d=d_period, smooth_k=smooth_k)
        
        result_df['stoch_k'] = stoch[f'STOCHk_{k_period}_{d_period}_{smooth_k}']
        result_df['stoch_d'] = stoch[f'STOCHd_{k_period}_{d_period}_{smooth_k}']
        
        return result_df
    
    def add_average_true_range(self, 
                              df: pd.DataFrame,
                              period: int = 14) -> pd.DataFrame:
        """
        添加平均真实范围(ATR)指标
        
        参数:
            df: 包含价格数据的DataFrame
            period: ATR周期
            
        返回:
            添加了ATR的DataFrame
        """
        result_df = df.copy()
        
        col_name = f'atr_{period}'
        result_df[col_name] = ta.atr(high=result_df['high'], low=result_df['low'], 
                                     close=result_df['close'], length=period)
        
        return result_df
    
    def add_on_balance_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加能量潮(OBV)指标
        
        参数:
            df: 包含价格和成交量数据的DataFrame
            
        返回:
            添加了OBV的DataFrame
        """
        result_df = df.copy()
        
        result_df['obv'] = ta.obv(close=result_df['close'], volume=result_df['volume'])
        
        return result_df
    
    def add_ichimoku_cloud(self, 
                          df: pd.DataFrame,
                          tenkan: int = 9,
                          kijun: int = 26,
                          senkou: int = 52) -> pd.DataFrame:
        """
        添加一目均衡表指标
        
        参数:
            df: 包含价格数据的DataFrame
            tenkan: 转换线周期
            kijun: 基准线周期
            senkou: 先行区间B周期
            
        返回:
            添加了一目均衡表的DataFrame
        """
        result_df = df.copy()
        
        ichimoku = ta.ichimoku(high=result_df['high'], low=result_df['low'], close=result_df['close'],
                              tenkan=tenkan, kijun=kijun, senkou=senkou)
        
        # 提取各个指标
        for col in ichimoku.columns:
            result_df[f'ichimoku_{col.lower()}'] = ichimoku[col]
        
        return result_df
    
    def add_fibonacci_retracement(self, 
                                 df: pd.DataFrame,
                                 period: int = 100) -> pd.DataFrame:
        """
        添加斐波那契回调水平
        
        参数:
            df: 包含价格数据的DataFrame
            period: 计算区间的周期
            
        返回:
            添加了斐波那契回调水平的DataFrame
        """
        result_df = df.copy()
        
        # 计算区间内的高点和低点
        result_df['period_high'] = result_df['high'].rolling(window=period).max()
        result_df['period_low'] = result_df['low'].rolling(window=period).min()
        
        # 计算斐波那契回调水平
        result_df['fib_diff'] = result_df['period_high'] - result_df['period_low']
        result_df['fib_23.6%'] = result_df['period_high'] - 0.236 * result_df['fib_diff']
        result_df['fib_38.2%'] = result_df['period_high'] - 0.382 * result_df['fib_diff']
        result_df['fib_50.0%'] = result_df['period_high'] - 0.500 * result_df['fib_diff']
        result_df['fib_61.8%'] = result_df['period_high'] - 0.618 * result_df['fib_diff']
        result_df['fib_78.6%'] = result_df['period_high'] - 0.786 * result_df['fib_diff']
        
        # 删除中间计算列
        result_df.drop(['period_high', 'period_low', 'fib_diff'], axis=1, inplace=True)
        
        return result_df
    
    def add_price_channels(self, 
                          df: pd.DataFrame,
                          period: int = 20) -> pd.DataFrame:
        """
        添加价格通道
        
        参数:
            df: 包含价格数据的DataFrame
            period: 通道周期
            
        返回:
            添加了价格通道的DataFrame
        """
        result_df = df.copy()
        
        result_df[f'upper_channel_{period}'] = result_df['high'].rolling(window=period).max()
        result_df[f'lower_channel_{period}'] = result_df['low'].rolling(window=period).min()
        result_df[f'middle_channel_{period}'] = (result_df[f'upper_channel_{period}'] + result_df[f'lower_channel_{period}']) / 2
        
        return result_df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加成交量相关指标
        
        参数:
            df: 包含价格和成交量数据的DataFrame
            
        返回:
            添加了成交量指标的DataFrame
        """
        result_df = df.copy()
        
        # 成交量变化率
        result_df['volume_change'] = result_df['volume'].pct_change() * 100
        
        # 成交量移动平均
        result_df['volume_ma_5'] = ta.sma(result_df['volume'], length=5)
        result_df['volume_ma_20'] = ta.sma(result_df['volume'], length=20)
        
        # 相对成交量
        result_df['rel_volume'] = result_df['volume'] / result_df['volume_ma_20']
        
        # 成交量趋势确认 (价格上涨/下跌与成交量的关系)
        result_df['price_up'] = (result_df['close'] > result_df['close'].shift(1)).astype(int)
        result_df['volume_price_trend'] = result_df['price_up'] * result_df['volume']
        
        return result_df
    
    def add_momentum_indicators(self, 
                               df: pd.DataFrame,
                               periods: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """
        添加动量相关指标
        
        参数:
            df: 包含价格数据的DataFrame
            periods: 动量计算周期列表
            
        返回:
            添加了动量指标的DataFrame
        """
        result_df = df.copy()
        
        # 价格动量
        for period in periods:
            col_name = f'momentum_{period}'
            result_df[col_name] = (result_df['close'] / result_df['close'].shift(period) - 1) * 100
            
        # 变化率
        for period in periods:
            col_name = f'roc_{period}'
            result_df[col_name] = ta.roc(result_df['close'], length=period)
            
        # 变化率百分比
        result_df['daily_return'] = result_df['close'].pct_change() * 100
        
        return result_df
    
    def add_volatility_indicators(self, 
                                 df: pd.DataFrame,
                                 periods: List[int] = [5, 21, 63]) -> pd.DataFrame:
        """
        添加波动率相关指标
        
        参数:
            df: 包含价格数据的DataFrame
            periods: 波动率计算周期列表
            
        返回:
            添加了波动率指标的DataFrame
        """
        result_df = df.copy()
        
        # 历史波动率 (收益率的标准差)
        result_df['daily_return'] = result_df['close'].pct_change()
        
        for period in periods:
            col_name = f'volatility_{period}'
            result_df[col_name] = result_df['daily_return'].rolling(window=period).std() * np.sqrt(252) * 100
            
        # 真实波动率
        result_df['true_range'] = ta.true_range(high=result_df['high'], low=result_df['low'], close=result_df['close'])
        
        # 波动率比率
        result_df['volatility_ratio_5_21'] = result_df['volatility_5'] / result_df['volatility_21']
        
        return result_df
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标
        
        参数:
            df: 包含价格和成交量数据的DataFrame
            
        返回:
            添加了所有技术指标的DataFrame
        """
        logger.info("正在计算所有技术指标...")
        result_df = df.copy()
        
        # 打印列名信息用于调试
        logger.info("列名及其类型:")
        for col in result_df.columns:
            logger.info(f"列名: {col}, 类型: {type(col)}")
        
        # 处理多级索引列名，只保留第一级
        if isinstance(result_df.columns, pd.MultiIndex):
            result_df.columns = result_df.columns.get_level_values(0)
        
        # 确保列名都是小写的
        result_df.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in result_df.columns]
        
        # 打印处理后的列名
        logger.info(f"处理后的列名: {result_df.columns.tolist()}")
        
        # 添加各类指标
        result_df = self.add_moving_averages(result_df)
        result_df = self.add_exponential_moving_averages(result_df)
        result_df = self.add_bollinger_bands(result_df)
        result_df = self.add_rsi(result_df)
        result_df = self.add_macd(result_df)
        result_df = self.add_stochastic_oscillator(result_df)
        result_df = self.add_average_true_range(result_df)
        result_df = self.add_on_balance_volume(result_df)
        result_df = self.add_price_channels(result_df)
        result_df = self.add_volume_indicators(result_df)
        result_df = self.add_momentum_indicators(result_df)
        result_df = self.add_volatility_indicators(result_df)
        
        # 添加时间特征
        if 'date' in result_df.columns:
            result_df = self.add_time_features(result_df)
        
        logger.info("完成所有技术指标的计算")
        return result_df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征
        
        参数:
            df: 包含日期列的DataFrame
            
        返回:
            添加了时间特征的DataFrame
        """
        result_df = df.copy()
        
        # 确保日期列是datetime类型
        if 'date' in result_df.columns and not pd.api.types.is_datetime64_any_dtype(result_df['date']):
            result_df['date'] = pd.to_datetime(result_df['date'])
        
        # 提取日期特征
        result_df['day_of_week'] = result_df['date'].dt.dayofweek
        result_df['day_of_month'] = result_df['date'].dt.day
        result_df['week_of_year'] = result_df['date'].dt.isocalendar().week
        result_df['month'] = result_df['date'].dt.month
        result_df['quarter'] = result_df['date'].dt.quarter
        result_df['year'] = result_df['date'].dt.year
        result_df['is_month_start'] = result_df['date'].dt.is_month_start.astype(int)
        result_df['is_month_end'] = result_df['date'].dt.is_month_end.astype(int)
        result_df['is_quarter_start'] = result_df['date'].dt.is_quarter_start.astype(int)
        result_df['is_quarter_end'] = result_df['date'].dt.is_quarter_end.astype(int)
        result_df['is_year_start'] = result_df['date'].dt.is_year_start.astype(int)
        result_df['is_year_end'] = result_df['date'].dt.is_year_end.astype(int)
        
        return result_df
    
    def process_stock_data(self, 
                          df: pd.DataFrame, 
                          selected_indicators: List[str] = None) -> pd.DataFrame:
        """
        处理股票数据，计算指定的技术指标
        
        参数:
            df: 包含价格和成交量数据的DataFrame
            selected_indicators: 需要计算的指标列表，如果为None则计算所有指标
            
        返回:
            添加了指定技术指标的DataFrame
        """
        result_df = df.copy()
        
        # 对列名进行标准化处理
        result_df.columns = [col.lower() for col in result_df.columns]
        
        # 确保有必要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 如果没有指定特定指标，就计算所有指标
        if selected_indicators is None:
            return self.calculate_all_indicators(result_df)
            
        # 计算指定的指标
        for indicator in selected_indicators:
            if indicator.startswith('ma_'):
                period = int(indicator.split('_')[1])
                result_df = self.add_moving_averages(result_df, [period])
            
            elif indicator.startswith('ema_'):
                period = int(indicator.split('_')[1])
                result_df = self.add_exponential_moving_averages(result_df, [period])
            
            elif indicator.startswith('bb_'):
                result_df = self.add_bollinger_bands(result_df)
            
            elif indicator.startswith('rsi_'):
                period = int(indicator.split('_')[1])
                result_df = self.add_rsi(result_df, [period])
            
            elif indicator in ['macd', 'macd_signal', 'macd_hist']:
                result_df = self.add_macd(result_df)
            
            elif indicator in ['stoch_k', 'stoch_d']:
                result_df = self.add_stochastic_oscillator(result_df)
            
            elif indicator.startswith('atr_'):
                period = int(indicator.split('_')[1])
                result_df = self.add_average_true_range(result_df, period)
            
            elif indicator == 'obv':
                result_df = self.add_on_balance_volume(result_df)
            
            elif any(indicator.startswith(prefix) for prefix in ['volume_', 'rel_volume']):
                result_df = self.add_volume_indicators(result_df)
            
            elif indicator.startswith('momentum_') or indicator.startswith('roc_'):
                period = int(indicator.split('_')[1])
                result_df = self.add_momentum_indicators(result_df, [period])
            
            elif indicator.startswith('volatility_'):
                period = int(indicator.split('_')[1])
                result_df = self.add_volatility_indicators(result_df, [period])
        
        # 添加时间特征
        if 'date' in result_df.columns and 'day_of_week' not in result_df.columns:
            result_df = self.add_time_features(result_df)
            
        return result_df
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        分析各个技术指标并生成趋势预测
        
        参数:
            df: 包含技术指标的DataFrame
            
        返回:
            包含分析结果的字典
        """
        analysis = {
            'signals': [],
            'score': 0,
            'max_score': 0,
            'details': {}
        }
        
        # 1. 移动平均线分析
        try:
            if df['ma_5'].iloc[-1] > df['ma_20'].iloc[-1]:
                analysis['signals'].append("短期均线在长期均线上方，可能呈现上升趋势")
                analysis['score'] += 1
            else:
                analysis['signals'].append("短期均线在长期均线下方，可能呈现下降趋势")
            analysis['max_score'] += 1
            
            # 计算均线趋势
            ma5_trend = df['ma_5'].iloc[-1] - df['ma_5'].iloc[-5]
            ma20_trend = df['ma_20'].iloc[-1] - df['ma_20'].iloc[-5]
            if ma5_trend > 0 and ma20_trend > 0:
                analysis['signals'].append("短期和长期均线都向上，强势上涨信号")
                analysis['score'] += 1
            analysis['max_score'] += 1
        except Exception as e:
            logger.warning(f"移动平均线分析发生错误: {str(e)}")

        # 2. MACD分析
        try:
            if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:
                analysis['signals'].append("MACD金叉，可能是买入信号")
                analysis['score'] += 1
            else:
                analysis['signals'].append("MACD死叉，可能是卖出信号")
            analysis['max_score'] += 1
            
            if df['macd_hist'].iloc[-1] > 0:
                analysis['signals'].append("MACD柱状图为正，上升趋势")
                analysis['score'] += 1
            analysis['max_score'] += 1
        except Exception as e:
            logger.warning(f"MACD分析发生错误: {str(e)}")

        # 3. RSI分析
        try:
            rsi_14 = df['rsi_14'].iloc[-1]
            if rsi_14 > 70:
                analysis['signals'].append("RSI超买，可能面临回调")
            elif rsi_14 < 30:
                analysis['signals'].append("RSI超卖，可能存在反弹机会")
            elif 40 <= rsi_14 <= 60:
                analysis['signals'].append("RSI处于中性区域")
                analysis['score'] += 1
            analysis['max_score'] += 1
        except Exception as e:
            logger.warning(f"RSI分析发生错误: {str(e)}")

        # 4. 布林带分析
        try:
            close = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            
            if close > bb_upper:
                analysis['signals'].append("价格突破布林带上轨，可能超买")
            elif close < bb_lower:
                analysis['signals'].append("价格突破布林带下轨，可能超卖")
            else:
                analysis['signals'].append("价格在布林带内运行，趋势平稳")
                analysis['score'] += 1
            analysis['max_score'] += 1
        except Exception as e:
            logger.warning(f"布林带分析发生错误: {str(e)}")

        # 5. 成交量分析
        try:
            if df['volume'].iloc[-1] > df['volume_ma_20'].iloc[-1]:
                analysis['signals'].append("成交量高于20日均量，交投活跃")
                analysis['score'] += 1
            analysis['max_score'] += 1
        except Exception as e:
            logger.warning(f"成交量分析发生错误: {str(e)}")

        # 6. 波动率分析
        try:
            if df['volatility_ratio_5_21'].iloc[-1] > 1.1:
                analysis['signals'].append("短期波动率显著高于长期，市场不稳定")
            elif df['volatility_ratio_5_21'].iloc[-1] < 0.9:
                analysis['signals'].append("市场波动率较低，可能酝酿大行情")
            else:
                analysis['signals'].append("波动率处于正常范围")
                analysis['score'] += 1
            analysis['max_score'] += 1
        except Exception as e:
            logger.warning(f"波动率分析发生错误: {str(e)}")

        # 计算综合得分
        analysis['score_percentage'] = (analysis['score'] / analysis['max_score']) * 100 if analysis['max_score'] > 0 else 0
        
        # 生成趋势判断
        if analysis['score_percentage'] >= 70:
            analysis['trend'] = "强势上涨"
            analysis['recommendation'] = "可以考虑买入"
        elif analysis['score_percentage'] >= 50:
            analysis['trend'] = "温和上涨"
            analysis['recommendation'] = "观望为主，可小仓位买入"
        elif analysis['score_percentage'] >= 30:
            analysis['trend'] = "震荡调整"
            analysis['recommendation'] = "建议观望"
        else:
            analysis['trend'] = "下跌趋势"
            analysis['recommendation'] = "建议离场观望"
            
        return analysis


# 使用示例
if __name__ == "__main__":
    import yfinance as yf
    
    try:
        # 获取某股票数据
        ticker = "AAPL"
        logger.info(f"开始下载 {ticker} 的股票数据...")
        data = yf.download(ticker, start="2025-01-01")
        
        if data.empty:
            logger.error(f"未能获取到 {ticker} 的数据")
            sys.exit(1)
            
        logger.info(f"\n原始数据预览:")
        logger.info("-" * 50)
        logger.info(f"数据形状: {data.shape}")
        logger.info(f"数据列名: {data.columns.tolist()}")
        logger.info(f"数据预览:\n{data.head()}")
        logger.info("-" * 50)
        
        # 初始化技术指标处理器
        processor = TechnicalIndicatorProcessor()
        
        # 计算所有技术指标
        logger.info("\n开始计算技术指标...")
        processed_data = processor.calculate_all_indicators(data)
        
        # 查看结果
        logger.info("\n计算完成! 技术指标统计信息:")
        logger.info("-" * 50)
        original_columns = len(data.columns)
        new_columns = len(processed_data.columns)
        logger.info(f"原始指标数量: {original_columns}")
        logger.info(f"新增指标数量: {new_columns - original_columns}")
        logger.info(f"总计指标数量: {new_columns}")
        
        # 按类别显示指标
        indicator_categories = {
            '移动平均线': [col for col in processed_data.columns if col.startswith(('ma_', 'ema_'))],
            '布林带': [col for col in processed_data.columns if col.startswith('bb_')],
            'RSI': [col for col in processed_data.columns if col.startswith('rsi_')],
            'MACD': [col for col in processed_data.columns if 'macd' in col],
            '成交量': [col for col in processed_data.columns if 'volume' in col],
            '波动率': [col for col in processed_data.columns if 'volatility' in col],
            '动量': [col for col in processed_data.columns if 'momentum' in col or 'roc' in col],
            '其他': []
        }
        
        logger.info("\n按类别显示指标:")
        for category, indicators in indicator_categories.items():
            logger.info(f"\n{category}:")
            for indicator in indicators:
                logger.info(f"  - {indicator}:")
                logger.info(f"    前5行数据:\n{processed_data[indicator].head()}")
                logger.info(f"    基本统计:\n{processed_data[indicator].describe()}\n")
        
        # 添加趋势分析
        logger.info("\n开始进行趋势分析...")
        analysis_result = processor.analyze_trend(processed_data)
        
        logger.info("\n=== 趋势分析报告 ===")
        logger.info("-" * 50)
        logger.info(f"综合得分: {analysis_result['score_percentage']:.2f}%")
        logger.info(f"趋势判断: {analysis_result['trend']}")
        logger.info(f"操作建议: {analysis_result['recommendation']}")
        logger.info("\n详细分析:")
        for signal in analysis_result['signals']:
            logger.info(f"- {signal}")
            
        # 预测下一个交易日的趋势
        last_close = processed_data['close'].iloc[-1]
        ma5_trend = processed_data['ma_5'].iloc[-1] - processed_data['ma_5'].iloc[-5]
        price_change_pred = ma5_trend * (analysis_result['score_percentage'] / 100)
        pred_price = last_close + price_change_pred
        
        logger.info("\n=== 预测信息 ===")
        logger.info("-" * 50)
        logger.info(f"当前收盘价: {last_close:.2f}")
        logger.info(f"预测下一交易日价格: {pred_price:.2f}")
        logger.info(f"预测涨跌幅: {(price_change_pred/last_close*100):.2f}%")
        
        # 保存结果到CSV文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, f"{ticker}_technical_indicators.csv")
        processed_data.to_csv(output_file)
        logger.info(f"\n结果已保存到文件: {output_file}")
        
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}")
        raise