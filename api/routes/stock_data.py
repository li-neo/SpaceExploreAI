#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据相关API路由
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field
import yfinance as yf

from ...data.finance.technical_indicators import TechnicalIndicatorProcessor
from .auth import User, get_current_active_user

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter(prefix="/api/stock-data", tags=["股票数据"])

# 加载技术指标处理器
indicator_processor = TechnicalIndicatorProcessor()

# 模型定义
class StockDataRequest(BaseModel):
    ticker: str
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)")
    include_indicators: bool = Field(False, description="是否包含技术指标")

class TickerInfo(BaseModel):
    ticker: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    country: Optional[str] = None

# 股票数据相关函数
def get_stock_data(ticker: str, start_date: str, end_date: str, include_indicators: bool = False):
    """获取股票历史数据"""
    try:
        # 使用yfinance下载数据
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"无法获取股票 {ticker} 的数据")
        
        # 重置索引
        data.reset_index(inplace=True)
        
        # 如果需要添加技术指标
        if include_indicators:
            # 为技术指标计算准备数据
            price_data = data.copy()
            price_data.set_index('Date', inplace=True)
            
            # 计算技术指标
            price_data = indicator_processor.calculate_all_indicators(price_data)
            
            # 重置索引
            price_data.reset_index(inplace=True)
            data = price_data
        
        # 转换为字典列表
        result = data.to_dict(orient='records')
        return result
    except Exception as e:
        logger.error(f"获取股票数据出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取股票数据出错: {str(e)}")

def get_ticker_info(ticker: str) -> TickerInfo:
    """获取股票基本信息"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return TickerInfo(
            ticker=ticker,
            name=info.get('longName', info.get('shortName', ticker)),
            sector=info.get('sector'),
            industry=info.get('industry'),
            market_cap=info.get('marketCap'),
            country=info.get('country')
        )
    except Exception as e:
        logger.error(f"获取股票信息出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取股票信息出错: {str(e)}")

# API端点
@router.post("/", response_model=List[Dict[str, Any]])
async def get_historical_data(
    request: StockDataRequest, 
    current_user: User = Depends(get_current_active_user)
):
    """获取股票历史数据"""
    return get_stock_data(
        request.ticker, 
        request.start_date, 
        request.end_date, 
        request.include_indicators
    )

@router.get("/{ticker}", response_model=List[Dict[str, Any]])
async def get_recent_data(
    ticker: str = Path(..., description="股票代码"),
    days: int = Query(30, description="天数", ge=1, le=365),
    include_indicators: bool = Query(False, description="是否包含技术指标"),
    current_user: User = Depends(get_current_active_user)
):
    """获取最近N天的股票数据"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    return get_stock_data(ticker, start_date, end_date, include_indicators)

@router.get("/{ticker}/info", response_model=TickerInfo)
async def get_stock_info(
    ticker: str = Path(..., description="股票代码"),
    current_user: User = Depends(get_current_active_user)
):
    """获取股票基本信息"""
    return get_ticker_info(ticker)

@router.get("/indicators/list")
async def list_available_indicators(current_user: User = Depends(get_current_active_user)):
    """列出可用的技术指标"""
    # 获取指标处理器支持的所有方法
    indicator_methods = [method for method in dir(indicator_processor) 
                         if method.startswith('add_') and callable(getattr(indicator_processor, method))]
    
    # 格式化结果
    indicators = []
    for method in indicator_methods:
        name = method.replace('add_', '')
        indicators.append({
            "id": name,
            "name": name.replace('_', ' ').title(),
            "description": getattr(indicator_processor, method).__doc__
        })
    
    return indicators 