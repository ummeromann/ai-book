import React from 'react';
import ChatBot from '../components/ChatBot';

// This component wraps the entire app and adds global features
export default function Root({ children }: { children: React.ReactNode }) {
  // Safe access to environment variables in browser
  const apiBaseUrl = typeof process !== 'undefined' && process.env?.REACT_APP_API_URL
    ? process.env.REACT_APP_API_URL
    : 'http://localhost:8000';

  return (
    <>
      {children}
      <ChatBot apiBaseUrl={apiBaseUrl} />
    </>
  );
}
