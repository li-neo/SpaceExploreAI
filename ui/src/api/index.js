/**
 * API服务模块
 * 封装所有与后端API的交互
 */

import axios from 'axios';

// 从环境变量获取API地址
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 创建axios实例
const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json'
  }
});

// 添加请求拦截器 - 在请求头中加入认证信息
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// 添加响应拦截器 - 处理错误和Token过期
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // 处理认证错误
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// 认证相关API
export const authAPI = {
  // 登录
  login: async (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    const response = await api.post('/token', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded'
      }
    });
    return response.data;
  },
  
  // 获取当前用户信息
  getCurrentUser: async () => {
    const response = await api.get('/users/me');
    return response.data;
  }
};

// 股票数据相关API
export const stockDataAPI = {
  // 获取股票历史数据
  getHistoricalData: async (ticker, startDate, endDate, includeIndicators = false) => {
    const response = await api.post('/api/stock-data', {
      ticker,
      start_date: startDate,
      end_date: endDate,
      include_indicators: includeIndicators
    });
    return response.data;
  },
  
  // 获取最近N天的股票数据
  getRecentData: async (ticker, days = 30, includeIndicators = false) => {
    const response = await api.get(`/api/stock-data/${ticker}`, {
      params: { days, include_indicators: includeIndicators }
    });
    return response.data;
  },
  
  // 获取股票基本信息
  getStockInfo: async (ticker) => {
    const response = await api.get(`/api/stock-data/${ticker}/info`);
    return response.data;
  },
  
  // 获取可用的技术指标列表
  getAvailableIndicators: async () => {
    const response = await api.get('/api/stock-data/indicators/list');
    return response.data;
  }
};

// 预测相关API
export const predictionAPI = {
  // 预测股票未来价格
  predictStock: async (ticker, days = 5, modelPath = null, confidenceThreshold = 0) => {
    const response = await api.post('/api/prediction', {
      ticker,
      days,
      model_path: modelPath,
      confidence_threshold: confidenceThreshold
    });
    return response.data;
  },
  
  // 批量预测多只股票
  batchPredict: async (tickers, days = 5, modelPath = null) => {
    const response = await api.post('/api/prediction/batch', {
      tickers,
      days,
      model_path: modelPath
    });
    return response.data;
  },
  
  // 获取可用模型列表
  getAvailableModels: async () => {
    const response = await api.get('/api/prediction/models');
    return response.data;
  }
};

export default {
  auth: authAPI,
  stockData: stockDataAPI,
  prediction: predictionAPI
}; 