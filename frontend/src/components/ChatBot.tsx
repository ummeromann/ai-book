import React, { useState, useEffect, useRef } from 'react';
import styles from './ChatBot.module.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: any[];
}

interface ChatBotProps {
  apiBaseUrl?: string;
}

const ChatBot: React.FC<ChatBotProps> = ({ apiBaseUrl = 'http://localhost:8000' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [selectedText, setSelectedText] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Listen for text selection
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection?.toString().trim();
      if (text && text.length > 0) {
        setSelectedText(text);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  const sendMessage = async (useSelectedText: boolean = false) => {
    if (!input.trim()) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
    };

    setMessages(prev => [...prev, userMessage]);
    setLoading(true);

    try {
      const endpoint = useSelectedText ? '/chat/selected' : '/chat';
      const payload = useSelectedText
        ? {
            query: input,
            selected_text: selectedText,
            session_id: sessionId,
          }
        : {
            query: input,
            session_id: sessionId,
            max_results: 5,
          };

      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Failed to get response from server');
      }

      const data = await response.json();

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setSessionId(data.session_id);
      setInput('');

      // Clear selected text after using it
      if (useSelectedText) {
        setSelectedText('');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSessionId(null);
    setSelectedText('');
  };

  return (
    <>
      {/* Floating Chat Button */}
      <button
        className={styles.floatingButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle chat"
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          {/* Header */}
          <div className={styles.chatHeader}>
            <h3>Physical AI Book Assistant</h3>
            <button onClick={clearChat} className={styles.clearButton}>
              Clear
            </button>
          </div>

          {/* Selected Text Indicator */}
          {selectedText && (
            <div className={styles.selectedTextBanner}>
              <span>📄 Text selected ({selectedText.length} chars)</span>
              <button
                onClick={() => setSelectedText('')}
                className={styles.clearSelectedText}
              >
                ✕
              </button>
            </div>
          )}

          {/* Messages */}
          <div className={styles.messagesContainer}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                <p>👋 Hi! I'm your AI assistant for the Physical AI book.</p>
                <p>Ask me anything about the book, or select text to ask specific questions!</p>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`${styles.message} ${
                  msg.role === 'user' ? styles.userMessage : styles.assistantMessage
                }`}
              >
                <div className={styles.messageContent}>
                  {msg.content}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className={styles.sources}>
                    <details>
                      <summary>Sources ({msg.sources.length})</summary>
                      <ul>
                        {msg.sources.map((source, sidx) => (
                          <li key={sidx}>
                            {source.module && source.chapter && (
                              <strong>
                                {source.module}/{source.chapter}
                              </strong>
                            )}
                            {source.text && (
                              <p className={styles.sourceText}>{source.text}</p>
                            )}
                            {source.score && (
                              <span className={styles.sourceScore}>
                                Score: {source.score.toFixed(3)}
                              </span>
                            )}
                          </li>
                        ))}
                      </ul>
                    </details>
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className={`${styles.message} ${styles.assistantMessage}`}>
                <div className={styles.loadingDots}>
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className={styles.inputContainer}>
            <textarea
              className={styles.input}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about the book..."
              rows={2}
              disabled={loading}
            />
            <div className={styles.buttonGroup}>
              <button
                className={styles.sendButton}
                onClick={() => sendMessage(false)}
                disabled={loading || !input.trim()}
              >
                Send
              </button>
              {selectedText && (
                <button
                  className={styles.selectedButton}
                  onClick={() => sendMessage(true)}
                  disabled={loading || !input.trim()}
                  title="Answer from selected text only"
                >
                  Ask Selected
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatBot;
