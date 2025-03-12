import React, { useState } from 'react';
import { Form, Input, Button, Card, Alert, Typography, Layout } from 'antd';
import { UserOutlined, LockOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { authAPI } from '../api';

const { Title } = Typography;
const { Content } = Layout;

/**
 * 登录页面组件
 */
const Login = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async (values) => {
    try {
      setLoading(true);
      setError(null);
      
      // 调用登录API
      const { username, password } = values;
      const data = await authAPI.login(username, password);
      
      // 保存认证信息
      localStorage.setItem('token', data.access_token);
      
      // 获取用户信息
      const user = await authAPI.getCurrentUser();
      localStorage.setItem('user', JSON.stringify(user));
      
      // 跳转到首页
      navigate('/dashboard');
    } catch (err) {
      console.error('登录失败:', err);
      setError(err.response?.data?.detail || '登录失败，请检查用户名和密码');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Content style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
        <Card style={{ width: 400, boxShadow: '0 4px 8px rgba(0,0,0,0.1)' }}>
          <div style={{ textAlign: 'center', marginBottom: 24 }}>
            <Title level={2}>股价预测系统</Title>
            <Title level={4} style={{ marginTop: 0 }}>用户登录</Title>
          </div>
          
          {error && (
            <Alert
              message="登录错误"
              description={error}
              type="error"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}
          
          <Form
            name="login"
            initialValues={{ remember: true }}
            onFinish={handleSubmit}
            autoComplete="off"
            size="large"
          >
            <Form.Item
              name="username"
              rules={[{ required: true, message: '请输入用户名' }]}
            >
              <Input 
                prefix={<UserOutlined />} 
                placeholder="用户名" 
              />
            </Form.Item>

            <Form.Item
              name="password"
              rules={[{ required: true, message: '请输入密码' }]}
            >
              <Input.Password 
                prefix={<LockOutlined />} 
                placeholder="密码" 
              />
            </Form.Item>

            <Form.Item>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={loading} 
                block
              >
                登录
              </Button>
            </Form.Item>
            
            <div style={{ textAlign: 'center' }}>
              <Typography.Text type="secondary">
                默认用户名: admin, 密码: admin123
              </Typography.Text>
            </div>
          </Form>
        </Card>
      </Content>
    </Layout>
  );
};

export default Login; 