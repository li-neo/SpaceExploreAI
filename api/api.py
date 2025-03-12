#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpaceExploreAI API 主模块

提供API接口的核心功能，由main.py调用
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建API路由基础
api_router = APIRouter()

# 身份验证
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@api_router.get("/info", tags=["API信息"])
async def api_info():
    """返回API版本和状态信息"""
    return {
        "name": "SpaceExploreAI API",
        "version": "1.0.0",
        "status": "active",
        "description": "股票预测AI系统API"
    }

@api_router.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

# 错误处理
@api_router.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """通用异常处理器"""
    logger.error(f"API错误: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"服务器内部错误: {str(exc)}"}
    ) 