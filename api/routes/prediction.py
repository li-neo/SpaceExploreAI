#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测相关API路由
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
import yfinance as yf
import torch

from ...train.trainer import create_stock_predictor_from_checkpoint
from ...data.technical_indicators import TechnicalIndicatorProcessor
from ...data.data_processor import StockDataProcessor
from .auth import User, get_current_active_user

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter(prefix="/api/prediction", tags=["预测"])

# 默认模型路径
DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "./models/stock_transformer_best.pt")

# 加载技术指标处理器
indicator_processor = TechnicalIndicatorProcessor()

# 预测器缓存
predictor_cache = {}

# 模型定义
class PredictionRequest(BaseModel):
    ticker: str
    days: int = Field(5, description="预测天数", ge=1, le=30)
    model_path: Optional[str] = None
    confidence_threshold: Optional[float] = Field(0.0, description="置信度阈值", ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    ticker: str
    prediction: List[float]
    prediction_dates: List[str]
    confidence: Optional[float] = None
    direction: str
    historical_data: Optional[List[Dict[str, Any]]] = None
    indicators_used: Optional[List[str]] = None

# 预测相关函数
def prepare_prediction_data(ticker: str, days: int = 5):
    """准备预测所需的数据"""
    try:
        # 计算开始日期，获取足够的历史数据
        end_date = datetime.now()
        # 获取200天的数据，确保有足够的历史数据计算技术指标
        start_date = end_date - timedelta(days=260)
        
        # 下载数据
        data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"无法获取股票 {ticker} 的数据")
        
        # 计算技术指标
        data = indicator_processor.calculate_all_indicators(data)
        
        # 准备预测日期
        last_date = data.index[-1]
        prediction_dates = []
        current_date = last_date + timedelta(days=1)
        
        # 生成未来日期，跳过周末
        while len(prediction_dates) < days:
            if current_date.weekday() < 5:  # 0-4是周一到周五
                prediction_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # 获取用于显示的历史数据
        historical_data = data.tail(30).reset_index().to_dict(orient='records')
        
        return data, prediction_dates, historical_data
    except Exception as e:
        logger.error(f"准备预测数据出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"准备预测数据出错: {str(e)}")

def get_predictor(model_path: str = DEFAULT_MODEL_PATH):
    """获取或加载预测模型"""
    try:
        # 如果模型已经加载，直接返回
        if model_path in predictor_cache:
            return predictor_cache[model_path]
        
        # 否则加载模型
        logger.info(f"加载模型: {model_path}")
        predictor = create_stock_predictor_from_checkpoint(model_path)
        
        # 缓存模型
        predictor_cache[model_path] = predictor
        
        return predictor
    except Exception as e:
        logger.error(f"加载模型出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"加载模型出错: {str(e)}")

def prepare_features(data, sequence_length):
    """准备模型输入特征"""
    # 获取最近的序列数据
    features = data.iloc[-sequence_length:].values
    
    # 转换为张量
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # 添加批次维度
    features_tensor = features_tensor.unsqueeze(0)
    
    return features_tensor

# API端点
@router.post("/", response_model=PredictionResponse)
async def predict_stock(
    request: PredictionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """预测股票未来价格"""
    # 准备数据
    data, prediction_dates, historical_data = prepare_prediction_data(request.ticker, request.days)
    
    # 加载预测模型
    model_path = request.model_path if request.model_path else DEFAULT_MODEL_PATH
    predictor = get_predictor(model_path)
    
    try:
        # 准备模型输入
        features = prepare_features(data, predictor.model.max_seq_len)
        
        # 进行预测
        with torch.no_grad():
            predictions = predictor.predict(features)
        
        # 转换为numpy数组
        predictions_np = predictions.cpu().numpy()
        
        # 计算置信度
        if predictions_np.std() != 0:
            confidence = float(abs(predictions_np.mean()) / predictions_np.std())
        else:
            confidence = None
        
        # 确定预测方向
        if predictions_np.mean() > 0:
            direction = "上涨"
        elif predictions_np.mean() < 0:
            direction = "下跌"
        else:
            direction = "持平"
        
        # 检查置信度阈值
        if confidence and request.confidence_threshold > 0 and confidence < request.confidence_threshold:
            logger.warning(f"预测置信度 {confidence} 低于阈值 {request.confidence_threshold}")
        
        # 获取使用的指标列表
        indicators_used = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'adj close']]
        
        # 构造响应
        response = PredictionResponse(
            ticker=request.ticker,
            prediction=predictions_np.flatten().tolist(),
            prediction_dates=prediction_dates,
            confidence=confidence,
            direction=direction,
            historical_data=historical_data,
            indicators_used=indicators_used
        )
        
        return response
    except Exception as e:
        logger.error(f"预测出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"预测出错: {str(e)}")

@router.get("/models")
async def list_available_models(current_user: User = Depends(get_current_active_user)):
    """列出可用的预测模型"""
    # 获取模型目录
    models_dir = os.path.dirname(DEFAULT_MODEL_PATH)
    
    try:
        # 列出所有.pt文件
        models = []
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pt'):
                    model_path = os.path.join(models_dir, file)
                    
                    # 获取文件信息
                    stat = os.stat(model_path)
                    created_time = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                    size_mb = stat.st_size / (1024 * 1024)
                    
                    models.append({
                        "name": file,
                        "path": model_path,
                        "created": created_time,
                        "size_mb": round(size_mb, 2),
                        "is_default": model_path == DEFAULT_MODEL_PATH
                    })
        
        return {"models": models}
    except Exception as e:
        logger.error(f"列出模型出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"列出模型出错: {str(e)}")

@router.post("/batch")
async def batch_predict(
    tickers: List[str] = Body(..., description="股票代码列表"),
    days: int = Body(5, description="预测天数", ge=1, le=30),
    model_path: Optional[str] = Body(None, description="模型路径"),
    current_user: User = Depends(get_current_active_user)
):
    """批量预测多只股票"""
    results = []
    errors = []
    
    for ticker in tickers:
        try:
            # 创建请求对象
            request = PredictionRequest(ticker=ticker, days=days, model_path=model_path)
            
            # 调用单股票预测
            prediction = await predict_stock(request, current_user)
            results.append(prediction)
        except HTTPException as e:
            errors.append({"ticker": ticker, "error": e.detail})
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
    
    return {"results": results, "errors": errors} 