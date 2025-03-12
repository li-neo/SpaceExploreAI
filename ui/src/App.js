import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { Layout, Menu, Button, Typography, Avatar, Dropdown } from 'antd';
import {
  DashboardOutlined,
  StockOutlined,
  BarChartOutlined,
  SettingOutlined,
  UserOutlined,
  LogoutOutlined,
} from '@ant-design/icons';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import { authAPI } from './api';

const { Header, Content, Footer, Sider } = Layout;
const { Title } = Typography;

/**
 * 需要认证的路由包装器
 */
const PrivateRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  const navigate = useNavigate();

  useEffect(() => {
    const verifyToken = async () => {
      try {
        // 验证token是否有效
        await authAPI.getCurrentUser();
      } catch (error) {
        // 无效token，重定向到登录页
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        navigate('/login');
      }
    };

    if (token) {
      verifyToken();
    }
  }, [navigate]);

  return token ? children : <Navigate to="/login" />;
};

/**
 * 主应用布局组件
 */
const AppLayout = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    // 尝试从localStorage获取用户信息
    const userStr = localStorage.getItem('user');
    if (userStr) {
      try {
        setUser(JSON.parse(userStr));
      } catch (e) {
        console.error('解析用户信息失败:', e);
      }
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  const userMenu = (
    <Menu>
      <Menu.Item key="profile" icon={<UserOutlined />}>
        个人资料
      </Menu.Item>
      <Menu.Item key="settings" icon={<SettingOutlined />}>
        设置
      </Menu.Item>
      <Menu.Divider />
      <Menu.Item key="logout" icon={<LogoutOutlined />} onClick={handleLogout}>
        退出登录
      </Menu.Item>
    </Menu>
  );

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        collapsible
        collapsed={collapsed}
        onCollapse={setCollapsed}
        theme="dark"
      >
        <div style={{ height: 32, margin: 16, textAlign: 'center' }}>
          <Title level={4} style={{ color: 'white', margin: 0 }}>
            股价预测系统
          </Title>
        </div>
        <Menu theme="dark" defaultSelectedKeys={['1']} mode="inline">
          <Menu.Item key="1" icon={<DashboardOutlined />} onClick={() => navigate('/dashboard')}>
            仪表盘
          </Menu.Item>
          <Menu.Item key="2" icon={<StockOutlined />} onClick={() => navigate('/stocks')}>
            股票数据
          </Menu.Item>
          <Menu.Item key="3" icon={<BarChartOutlined />} onClick={() => navigate('/predictions')}>
            预测分析
          </Menu.Item>
        </Menu>
      </Sider>
      <Layout className="site-layout">
        <Header style={{ padding: '0 16px', background: '#fff', display: 'flex', justifyContent: 'flex-end', alignItems: 'center' }}>
          {user && (
            <Dropdown overlay={userMenu} placement="bottomRight">
              <div style={{ cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
                <Avatar icon={<UserOutlined />} />
                <span style={{ marginLeft: 8 }}>{user.username}</span>
              </div>
            </Dropdown>
          )}
        </Header>
        <Content style={{ margin: '0 16px' }}>
          {children}
        </Content>
        <Footer style={{ textAlign: 'center' }}>
          SpaceExploreAI 股价预测系统 ©{new Date().getFullYear()} 深思未来科技
        </Footer>
      </Layout>
    </Layout>
  );
};

/**
 * 主应用组件
 */
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <AppLayout>
                <Dashboard />
              </AppLayout>
            </PrivateRoute>
          }
        />
        <Route
          path="/stocks"
          element={
            <PrivateRoute>
              <AppLayout>
                <div>股票数据页面（待实现）</div>
              </AppLayout>
            </PrivateRoute>
          }
        />
        <Route
          path="/predictions"
          element={
            <PrivateRoute>
              <AppLayout>
                <div>预测分析页面（待实现）</div>
              </AppLayout>
            </PrivateRoute>
          }
        />
        <Route path="/" element={<Navigate to="/dashboard" />} />
      </Routes>
    </Router>
  );
}

export default App; 