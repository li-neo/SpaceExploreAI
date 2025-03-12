 # SpaceExploreAI API 服务

这是SpaceExploreAI项目的RESTful API服务，用于提供股票数据查询、模型训练和预测服务。

## 功能

- 用户认证
- 股票历史数据获取和处理
- 技术指标计算
- 股价预测服务
- 模型管理

## 安装

确保已安装Python 3.8+和pip：

```bash
# 安装依赖
pip install -r requirements.txt
```

## 启动服务

### 开发环境

```bash
uvicorn SpaceExploreAI.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产环境

```bash
# 使用Gunicorn作为WSGI服务器（生产环境）
gunicorn -w 4 -k uvicorn.workers.UvicornWorker SpaceExploreAI.api.main:app --bind 0.0.0.0:8000
```

## API文档

启动服务后，可以在以下地址访问自动生成的API文档：

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## 环境变量

可以通过以下环境变量配置API服务：

- `API_SECRET_KEY`: 用于JWT认证的密钥
- `MODEL_PATH`: 默认模型路径

示例：

```bash
export API_SECRET_KEY=your-secret-key
export MODEL_PATH=./models/stock_transformer_best.pt
```

## 路由结构

API服务的主要路由结构：

### 认证路由

- `POST /token`: 获取访问令牌
- `GET /users/me`: 获取当前用户信息

### 股票数据路由

- `POST /api/stock-data`: 获取股票历史数据
- `GET /api/stock-data/{ticker}`: 获取最近N天的股票数据
- `GET /api/stock-data/{ticker}/info`: 获取股票基本信息
- `GET /api/stock-data/indicators/list`: 列出可用的技术指标

### 预测路由

- `POST /api/prediction`: 预测股票未来价格
- `GET /api/prediction/models`: 列出可用的预测模型
- `POST /api/prediction/batch`: 批量预测多只股票

## 开发指南

### 添加新路由

1. 在`SpaceExploreAI/api/routes/`目录下创建新的路由模块
2. 在`SpaceExploreAI/api/main.py`中导入并注册新路由

```python
from .routes import new_route
app.include_router(new_route.router)
```

### 自定义认证

默认使用基于JWT的认证。如需更改，请修改`SpaceExploreAI/api/routes/auth.py`文件。

## 部署

### Docker部署

```bash
# 构建Docker镜像
docker build -t spaceexploreai-api .

# 运行容器
docker run -p 8000:8000 -e API_SECRET_KEY=your-secret-key spaceexploreai-api
```

### Kubernetes部署

请参考`deployment/k8s/`目录下的Kubernetes配置文件。