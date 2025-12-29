import React from 'react';
import ChatBot from '../components/ChatBot';
import AuthButton from '../components/Auth/AuthButton';
import { AuthProvider } from '../context/AuthContext';
import ExecutionEnvironment from '@docusaurus/ExecutionEnvironment';

// Get API URL safely - only access process.env during build time
const getApiUrl = () => {
  if (ExecutionEnvironment.canUseDOM) {
    // Client-side: use window location or default to localhost
    return 'http://localhost:8000';
  }
  return 'http://localhost:8000';
};

// This component wraps the entire app and adds global features
export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      {children}
      <ChatBot apiBaseUrl={getApiUrl()} />
      <div style={{
        position: 'fixed',
        top: '70px',
        right: '20px',
        zIndex: 1000
      }}>
        <AuthButton />
      </div>
    </AuthProvider>
  );
}
