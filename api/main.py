#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpaceExploreAI API 主入口

启动命令:
    uvicorn SpaceExploreAI.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import auth, stock_data, prediction

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="SpaceExploreAI 股价预测API",
    description="提供股票数据查询、模型训练和预测服务的API接口",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加路由
app.include_router(auth.router)
app.include_router(stock_data.router)
app.include_router(prediction.router)

@app.get("/", tags=["根路径"])
async def root():
    """API根路径，提供基本信息"""
    return {
        "name": "SpaceExploreAI 股价预测API",
        "version": "1.0.0",
        "description": "股票预测Transformer模型API服务",
        "docs": "/api/docs",
        "status": "running"
    }

@app.get("/health", tags=["健康检查"])
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

# 主入口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 