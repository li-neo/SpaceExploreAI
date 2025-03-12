# SpaceExploreAI Web UI

这是SpaceExploreAI项目的Web用户界面，使用React和Ant Design构建。

## 功能

- 股票数据可视化
- 实时股价预测
- 技术指标分析
- 批量股票分析
- 用户认证和权限管理

## 安装

确保已安装Node.js (v14+)和npm (v6+)：

```bash
# 安装依赖
npm install

# 启动开发服务器
npm start

# 构建生产版本
npm run build
```

## 开发指南

### 项目结构

```
ui/
├── public/                 # 静态资源
├── src/                    # 源代码
│   ├── api/                # API 接口定义
│   ├── components/         # 可复用组件
│   ├── pages/              # 页面组件
│   ├── contexts/           # React 上下文
│   ├── hooks/              # 自定义 hooks
│   ├── utils/              # 工具函数
│   ├── App.js              # 应用入口
│   └── index.js            # 渲染入口
├── package.json            # 项目配置和依赖
└── README.md               # 项目说明
```

### 开发注意事项

1. 所有API请求应使用`src/api`中定义的函数，不应直接在组件中调用API
2. 遵循React函数组件和Hooks的最佳实践
3. 使用Ant Design组件库来保持UI一致性
4. 所有页面路由应在`App.js`中定义

## 配置

在`.env`文件中配置API地址：

```
REACT_APP_API_URL=http://localhost:8000
```

## 部署

构建后的静态文件可以托管在任何支持静态网站的服务器上：

```bash
# 构建项目
npm run build

# 构建的文件在build目录，可以将其部署到Web服务器
``` 