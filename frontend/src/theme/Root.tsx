import React from 'react';
import ChatBot from '../components/ChatBot';

// This component wraps the entire app and adds global features
export default function Root({ children }: { children: React.ReactNode }) {
  return (
    <>
      {children}
      <ChatBot apiBaseUrl={process.env.REACT_APP_API_URL || 'http://localhost:8000'} />
    </>
  );
}
