#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
趋势反转策略参数配置
--------------------------------
此模块定义了趋势反转策略的所有可配置参数。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

@dataclass
class TrendArgs:
    """趋势反转策略参数配置"""
    
    # 回测参数
    backtest_start: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=30))
    backtest_end: datetime = field(default_factory=lambda: datetime.now())
    timeframes: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("1m", "1d"),   # 1分钟K线，1天数据（由于数据限制，1分钟数据最多只能获取1周）
        ("5m", "5d"),   # 5分钟K线，5天数据
        ("15m", "5d"),  # 15分钟K线，5天数据
        ("1h", "30d")   # 1小时K线，30天数据
    ])
    
    # 股票列表参数
    tickers: List[str] = field(default_factory=lambda: [
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
    ])
    reference_ticker: str = "QQQ"  # 参考指数
    
    # 背离分析参数
    divergence_lookback: int = 20  # 背离分析回看周期
    divergence_sensitivity: float = 1.0  # 背离检测敏感度
    min_price_change: float = 0.0  # 最小价格变化阈值
    min_indicator_change: float = 0.0  # 最小指标变化阈值
    
    # 成交量分析参数
    volume_pivot_bars: int = 5  # 枢轴点周围的K线数量
    volume_lookback: int = 30  # 成交量分析回看周期
    min_volume_ratio: float = 3.0  # 最小成交量比率
    high_volume_threshold: float = 2.0  # 高成交量阈值
    
    # 相对强弱分析参数
    rs_lookback: int = 1  # 相对强弱分析回看天数
    rs_threshold_bullish: float = 1.05  # 看涨相对强弱阈值
    rs_threshold_bearish: float = 0.95  # 看跌相对强弱阈值
    
    # 支撑阻力分析参数
    sr_lookback: int = 100  # 支撑阻力分析回看周期
    sr_price_threshold: float = 0.05  # 价格水平阈值
    sr_cluster_threshold: float = 0.01  # 聚集水平阈值
    
    # 信号强度阈值
    warning_threshold: int = 11  # 警告信号阈值 (0-14)
    min_signal_strength: int = 8  # 最小信号强度
    
    # 技术指标参数
    rsi_length: int = 14  # RSI周期
    macd_fast: int = 12  # MACD快线
    macd_slow: int = 26  # MACD慢线
    macd_signal: int = 9  # MACD信号线
    stoch_k: int = 5  # 随机指标K值
    stoch_d: int = 3  # 随机指标D值
    stoch_smooth: int = 3  # 随机指标平滑值
    
    # 移动平均线参数
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    
    # 调试参数
    debug_mode: bool = False  # 是否启用调试模式
    verbose: bool = True  # 是否打印详细信息
    plot_charts: bool = True  # 是否绘制图表
    save_results: bool = True  # 是否保存结果
    
    def to_dict(self) -> Dict:
        """将参数转换为字典格式"""
        return {
            "backtest": {
                "start": self.backtest_start.strftime("%Y-%m-%d"),
                "end": self.backtest_end.strftime("%Y-%m-%d"),
                "timeframes": self.timeframes
            },
            "tickers": {
                "symbols": self.tickers,
                "reference": self.reference_ticker
            },
            "divergence": {
                "lookback": self.divergence_lookback,
                "sensitivity": self.divergence_sensitivity,
                "min_price_change": self.min_price_change,
                "min_indicator_change": self.min_indicator_change
            },
            "volume": {
                "pivot_bars": self.volume_pivot_bars,
                "lookback": self.volume_lookback,
                "min_ratio": self.min_volume_ratio,
                "high_threshold": self.high_volume_threshold
            },
            "relative_strength": {
                "lookback": self.rs_lookback,
                "threshold_bullish": self.rs_threshold_bullish,
                "threshold_bearish": self.rs_threshold_bearish
            },
            "support_resistance": {
                "lookback": self.sr_lookback,
                "price_threshold": self.sr_price_threshold,
                "cluster_threshold": self.sr_cluster_threshold
            },
            "signal": {
                "warning_threshold": self.warning_threshold,
                "min_strength": self.min_signal_strength
            },
            "indicators": {
                "rsi": {"length": self.rsi_length},
                "macd": {
                    "fast": self.macd_fast,
                    "slow": self.macd_slow,
                    "signal": self.macd_signal
                },
                "stoch": {
                    "k": self.stoch_k,
                    "d": self.stoch_d,
                    "smooth": self.stoch_smooth
                },
                "sma": {"periods": self.sma_periods}
            },
            "debug": {
                "mode": self.debug_mode,
                "verbose": self.verbose,
                "plot_charts": self.plot_charts,
                "save_results": self.save_results
            }
        }
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'TrendArgs':
        """从字典创建参数实例"""
        args = cls()
        
        # 处理回测参数
        if "backtest_start" in config and "backtest_end" in config:
            args.backtest_start = datetime.strptime(config["backtest_start"], "%Y-%m-%d")
            args.backtest_end = datetime.strptime(config["backtest_end"], "%Y-%m-%d")
        elif "backtest" in config:
            args.backtest_start = datetime.strptime(config["backtest"]["start"], "%Y-%m-%d")
            args.backtest_end = datetime.strptime(config["backtest"]["end"], "%Y-%m-%d")
            if "timeframes" in config["backtest"]:
                args.timeframes = config["backtest"]["timeframes"]
        
        # 处理时间周期
        if "timeframes" in config:
            args.timeframes = config["timeframes"]
        
        # 处理股票列表
        if "tickers" in config:
            # 检查 tickers 是列表还是字典
            if isinstance(config["tickers"], list):
                args.tickers = config["tickers"]
            elif isinstance(config["tickers"], dict) and "symbols" in config["tickers"]:
                args.tickers = config["tickers"]["symbols"]
        
        # 处理参考股票
        if "reference_ticker" in config:
            args.reference_ticker = config["reference_ticker"]
        elif "tickers" in config and isinstance(config["tickers"], dict) and "reference" in config["tickers"]:
            args.reference_ticker = config["tickers"]["reference"]
        
        # 处理信号阈值
        if "min_signal_strength" in config:
            args.min_signal_strength = config["min_signal_strength"]
        elif "signal" in config and "min_strength" in config["signal"]:
            args.min_signal_strength = config["signal"]["min_strength"]
            
        if "warning_threshold" in config:
            args.warning_threshold = config["warning_threshold"]
        elif "signal" in config and "warning_threshold" in config["signal"]:
            args.warning_threshold = config["signal"]["warning_threshold"]
        
        # 处理背离参数
        if "divergence_sensitivity" in config:
            args.divergence_sensitivity = config["divergence_sensitivity"]
        elif "divergence" in config and "sensitivity" in config["divergence"]:
            args.divergence_sensitivity = config["divergence"]["sensitivity"]
            args.divergence_lookback = config["divergence"].get("lookback", args.divergence_lookback)
            args.min_price_change = config["divergence"].get("min_price_change", args.min_price_change)
            args.min_indicator_change = config["divergence"].get("min_indicator_change", args.min_indicator_change)
        
        # 处理成交量参数
        if "volume" in config:
            args.volume_pivot_bars = config["volume"].get("pivot_bars", args.volume_pivot_bars)
            args.volume_lookback = config["volume"].get("lookback", args.volume_lookback)
            args.min_volume_ratio = config["volume"].get("min_ratio", args.min_volume_ratio)
            args.high_volume_threshold = config["volume"].get("high_threshold", args.high_volume_threshold)
        
        # 处理相对强弱参数
        if "relative_strength" in config:
            args.rs_lookback = config["relative_strength"].get("lookback", args.rs_lookback)
            args.rs_threshold_bullish = config["relative_strength"].get("threshold_bullish", args.rs_threshold_bullish)
            args.rs_threshold_bearish = config["relative_strength"].get("threshold_bearish", args.rs_threshold_bearish)
        
        # 处理支撑阻力参数
        if "support_resistance" in config:
            args.sr_lookback = config["support_resistance"].get("lookback", args.sr_lookback)
            args.sr_price_threshold = config["support_resistance"].get("price_threshold", args.sr_price_threshold)
            args.sr_cluster_threshold = config["support_resistance"].get("cluster_threshold", args.sr_cluster_threshold)
        
        # 处理技术指标参数
        if "indicators" in config:
            if "rsi" in config["indicators"]:
                args.rsi_length = config["indicators"]["rsi"].get("length", args.rsi_length)
            if "macd" in config["indicators"]:
                args.macd_fast = config["indicators"]["macd"].get("fast", args.macd_fast)
                args.macd_slow = config["indicators"]["macd"].get("slow", args.macd_slow)
                args.macd_signal = config["indicators"]["macd"].get("signal", args.macd_signal)
            if "stoch" in config["indicators"]:
                args.stoch_k = config["indicators"]["stoch"].get("k", args.stoch_k)
                args.stoch_d = config["indicators"]["stoch"].get("d", args.stoch_d)
                args.stoch_smooth = config["indicators"]["stoch"].get("smooth", args.stoch_smooth)
            if "sma" in config["indicators"]:
                args.sma_periods = config["indicators"]["sma"].get("periods", args.sma_periods)
        
        # 处理调试参数
        if "debug_mode" in config:
            args.debug_mode = config["debug_mode"]
        elif "debug" in config and "mode" in config["debug"]:
            args.debug_mode = config["debug"]["mode"]
            
        if "verbose" in config:
            args.verbose = config["verbose"]
        elif "debug" in config and "verbose" in config["debug"]:
            args.verbose = config["debug"]["verbose"]
            
        if "plot_charts" in config:
            args.plot_charts = config["plot_charts"]
        elif "debug" in config and "plot_charts" in config["debug"]:
            args.plot_charts = config["debug"]["plot_charts"]
            
        if "save_results" in config:
            args.save_results = config["save_results"]
        elif "debug" in config and "save_results" in config["debug"]:
            args.save_results = config["debug"]["save_results"]
        
        return args
    
    def save_to_file(self, filepath: str) -> None:
        """保存参数到JSON文件"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrendArgs':
        """从JSON文件加载参数"""
        import json
        with open(filepath, 'r') as f:
            config = json.load(f)
        return cls.from_dict(config) 