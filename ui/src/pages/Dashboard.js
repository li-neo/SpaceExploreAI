import React, { useState, useEffect } from 'react';
import { 
  Layout, Card, Row, Col, Typography, Statistic, Select, Spin, 
  Button, DatePicker, Switch, message, Space, Divider, Badge 
} from 'antd';
import { 
  LineChartOutlined, StockOutlined, ArrowUpOutlined, 
  ArrowDownOutlined, DashboardOutlined, CalculatorOutlined 
} from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import dayjs from 'dayjs';
import { stockDataAPI, predictionAPI } from '../api';

const { Title, Text } = Typography;
const { Content } = Layout;
const { Option } = Select;
const { RangePicker } = DatePicker;

/**
 * 主仪表盘页面
 */
const Dashboard = () => {
  // 状态
  const [loading, setLoading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [ticker, setTicker] = useState('AAPL');
  const [stockData, setStockData] = useState([]);
  const [stockInfo, setStockInfo] = useState(null);
  const [dateRange, setDateRange] = useState([
    dayjs().subtract(90, 'day'),
    dayjs()
  ]);
  const [includeIndicators, setIncludeIndicators] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [priceChartOption, setPriceChartOption] = useState({});
  const [predictionChartOption, setPredictionChartOption] = useState({});
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);
  const [indicators, setIndicators] = useState([]);
  const [selectedIndicators, setSelectedIndicators] = useState(['ma_50', 'ma_200', 'rsi_14']);

  // 常用股票列表
  const commonStocks = [
    { value: 'AAPL', label: 'Apple' },
    { value: 'MSFT', label: 'Microsoft' },
    { value: 'GOOGL', label: 'Google' },
    { value: 'AMZN', label: 'Amazon' },
    { value: 'META', label: 'Meta' },
    { value: 'TSLA', label: 'Tesla' },
    { value: 'NVDA', label: 'NVIDIA' },
    { value: 'JPM', label: 'JPMorgan' },
    { value: 'V', label: 'Visa' },
    { value: 'WMT', label: 'Walmart' }
  ];

  // 加载股票数据
  const loadStockData = async () => {
    try {
      setLoading(true);
      const [startDate, endDate] = dateRange;
      
      // 获取股票数据
      const data = await stockDataAPI.getHistoricalData(
        ticker,
        startDate.format('YYYY-MM-DD'),
        endDate.format('YYYY-MM-DD'),
        includeIndicators
      );
      
      setStockData(data);
      
      // 获取股票信息
      const info = await stockDataAPI.getStockInfo(ticker);
      setStockInfo(info);
      
      // 更新图表
      updatePriceChart(data);
    } catch (error) {
      console.error('加载股票数据失败:', error);
      message.error('加载股票数据失败');
    } finally {
      setLoading(false);
    }
  };

  // 加载可用模型
  const loadModels = async () => {
    try {
      const { models } = await predictionAPI.getAvailableModels();
      setModels(models);
      
      // 设置默认模型
      const defaultModel = models.find(m => m.is_default) || models[0];
      if (defaultModel) {
        setSelectedModel(defaultModel.path);
      }
    } catch (error) {
      console.error('加载模型列表失败:', error);
    }
  };

  // 加载可用指标
  const loadIndicators = async () => {
    try {
      const indicators = await stockDataAPI.getAvailableIndicators();
      setIndicators(indicators);
    } catch (error) {
      console.error('加载指标列表失败:', error);
    }
  };

  // 初始化
  useEffect(() => {
    loadModels();
    loadIndicators();
  }, []);

  // 股票或日期变化时重新加载数据
  useEffect(() => {
    if (ticker && dateRange) {
      loadStockData();
    }
  }, [ticker, dateRange, includeIndicators]);

  // 更新价格图表
  const updatePriceChart = (data) => {
    // 准备数据
    const dates = data.map(item => item.Date);
    const prices = data.map(item => item.close);
    
    // 准备指标数据
    let indicatorSeries = [];
    if (includeIndicators) {
      // 为每个选中的指标创建系列
      selectedIndicators.forEach(indicator => {
        if (data[0] && data[0][indicator] !== undefined) {
          indicatorSeries.push({
            name: indicator,
            type: 'line',
            data: data.map(item => item[indicator]),
            smooth: true
          });
        }
      });
    }
    
    // 设置图表选项
    const option = {
      title: {
        text: `${ticker} 股价历史`,
        left: 'center'
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: ['收盘价', ...selectedIndicators],
        bottom: 0
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: dates
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      series: [
        {
          name: '收盘价',
          type: 'line',
          data: prices,
          smooth: true,
          lineStyle: {
            width: 3
          },
          itemStyle: {
            color: '#1890ff'
          }
        },
        ...indicatorSeries
      ]
    };
    
    setPriceChartOption(option);
  };

  // 预测股价
  const predictStock = async () => {
    try {
      setPredicting(true);
      
      // 调用预测API
      const result = await predictionAPI.predictStock(
        ticker,
        5,  // 预测5天
        selectedModel
      );
      
      setPrediction(result);
      
      // 更新预测图表
      updatePredictionChart(result);
      
      message.success('预测完成');
    } catch (error) {
      console.error('预测失败:', error);
      message.error('预测失败');
    } finally {
      setPredicting(false);
    }
  };

  // 更新预测图表
  const updatePredictionChart = (predictionData) => {
    // 获取最近30天的历史数据
    const recentData = stockData.slice(-30);
    const historicalDates = recentData.map(item => item.Date);
    const historicalPrices = recentData.map(item => item.close);
    
    // 准备预测数据
    const predictionDates = predictionData.prediction_dates;
    const predictionPrices = predictionData.prediction;
    
    // 合并日期和价格
    const allDates = [...historicalDates, ...predictionDates];
    
    // 创建历史价格系列，预测位置为null
    const historicalSeries = [
      ...historicalPrices,
      ...Array(predictionDates.length).fill(null)
    ];
    
    // 创建预测价格系列，历史位置为null
    const predictionSeries = [
      ...Array(historicalDates.length).fill(null),
      ...predictionPrices
    ];
    
    // 设置图表选项
    const option = {
      title: {
        text: `${ticker} 股价预测 (未来5天)`,
        left: 'center'
      },
      tooltip: {
        trigger: 'axis'
      },
      legend: {
        data: ['历史价格', '预测价格'],
        bottom: 0
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '15%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: allDates
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      series: [
        {
          name: '历史价格',
          type: 'line',
          data: historicalSeries,
          smooth: true,
          lineStyle: {
            width: 2
          },
          itemStyle: {
            color: '#1890ff'
          }
        },
        {
          name: '预测价格',
          type: 'line',
          data: predictionSeries,
          smooth: true,
          lineStyle: {
            width: 3,
            type: 'dashed'
          },
          itemStyle: {
            color: '#52c41a'
          }
        }
      ]
    };
    
    setPredictionChartOption(option);
  };

  // 渲染股票信息卡片
  const renderStockInfoCard = () => {
    if (!stockInfo) return null;
    
    return (
      <Card>
        <Statistic
          title={stockInfo.name}
          value={stockData.length > 0 ? stockData[stockData.length - 1].close : 0}
          precision={2}
          valueStyle={{ color: '#1890ff' }}
          prefix={<StockOutlined />}
          suffix="$"
        />
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">行业: {stockInfo.industry || '未知'}</Text>
          <br />
          <Text type="secondary">国家: {stockInfo.country || '未知'}</Text>
        </div>
      </Card>
    );
  };

  // 渲染预测结果卡片
  const renderPredictionCard = () => {
    if (!prediction) return null;
    
    const isUp = prediction.direction === '上涨';
    const color = isUp ? '#52c41a' : '#f5222d';
    const icon = isUp ? <ArrowUpOutlined /> : <ArrowDownOutlined />;
    
    return (
      <Card>
        <Statistic
          title="预测方向"
          value={prediction.direction}
          valueStyle={{ color }}
          prefix={icon}
        />
        {prediction.confidence !== null && (
          <Statistic
            title="置信度"
            value={prediction.confidence}
            precision={2}
            valueStyle={{ color: '#722ed1' }}
            prefix={<CalculatorOutlined />}
            suffix="%"
            style={{ marginTop: 16 }}
          />
        )}
      </Card>
    );
  };

  return (
    <Content style={{ padding: '20px' }}>
      <Title level={2}>
        <DashboardOutlined /> 股票分析仪表盘
      </Title>
      
      <Row gutter={[16, 16]}>
        <Col span={6}>
          <Card title="股票选择">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Select
                showSearch
                style={{ width: '100%' }}
                placeholder="选择股票"
                value={ticker}
                onChange={setTicker}
                optionFilterProp="label"
                options={commonStocks}
              />
              
              <RangePicker
                style={{ width: '100%' }}
                value={dateRange}
                onChange={setDateRange}
              />
              
              <div>
                <Text>包含技术指标:</Text>
                <Switch
                  checked={includeIndicators}
                  onChange={setIncludeIndicators}
                  style={{ marginLeft: 8 }}
                />
              </div>
              
              {includeIndicators && (
                <Select
                  mode="multiple"
                  style={{ width: '100%' }}
                  placeholder="选择技术指标"
                  value={selectedIndicators}
                  onChange={setSelectedIndicators}
                  options={indicators.map(ind => ({ label: ind.name, value: ind.id }))}
                />
              )}
              
              <Divider />
              
              <Select
                style={{ width: '100%' }}
                placeholder="选择预测模型"
                value={selectedModel}
                onChange={setSelectedModel}
                options={models.map(model => ({ label: model.name, value: model.path }))}
              />
              
              <Button
                type="primary"
                icon={<LineChartOutlined />}
                loading={predicting}
                onClick={predictStock}
                block
              >
                预测股价
              </Button>
            </Space>
          </Card>
        </Col>
        
        <Col span={6}>
          {renderStockInfoCard()}
        </Col>
        
        <Col span={6}>
          {renderPredictionCard()}
        </Col>
        
        <Col span={24}>
          <Card title="股价历史">
            <Spin spinning={loading}>
              <ReactECharts 
                option={priceChartOption} 
                style={{ height: '400px' }} 
              />
            </Spin>
          </Card>
        </Col>
        
        {prediction && (
          <Col span={24}>
            <Card title="股价预测">
              <ReactECharts 
                option={predictionChartOption} 
                style={{ height: '400px' }} 
              />
            </Card>
          </Col>
        )}
      </Row>
    </Content>
  );
};

export default Dashboard; 