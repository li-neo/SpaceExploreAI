#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
快速股票预测脚本
使用简化的接口进行单次预测
"""

import os
import sys
import argparse
from typing import List

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.inferencer import StockPredictor

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速股票预测")
    parser.add_argument("tickers", nargs="+", help="要预测的股票代码，例如：AAPL MSFT GOOG")
    args = parser.parse_args()
    
    # 创建预测器
    predictor = StockPredictor()
    
    # 打印标题
    print("\n==== 股票预测结果 ====\n")
    
    # 对每个股票进行预测
    for ticker in args.tickers:
        print(f"预测股票: {ticker}")
        result = predictor.predict(ticker)
        
        if "error" in result:
            print(f"  错误: {result['error']}")
        else:
            print(f"  预测收益率: {result['prediction']:.6f}")
            print(f"  预测时间: {result['timestamp']}")
        
        print("")
    
    print("==== 预测完成 ====\n")

if __name__ == "__main__":
    main() 