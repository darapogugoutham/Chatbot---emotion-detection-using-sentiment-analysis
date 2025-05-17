// src/App.js
import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    try {
      setIsLoading(true);
      
      // Add user message
      setMessages(prev => [...prev, {
        text: inputMessage,
        isBot: false,
        emotion: '',
        source: 'user'
      }]);

      // Get bot response
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage }),
      });

      if (!response.ok) throw new Error('API request failed');
      
      const data = await response.json();

      // Add bot message
      setMessages(prev => [...prev, {
        text: data.response,
        isBot: true,
        emotion: data.emotion,
        source: data.source
      }]);

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        text: "Sorry, I'm having trouble responding right now.",
        isBot: true,
        emotion: 'error',
        source: 'system'
      }]);
    } finally {
      setInputMessage('');
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.isBot ? 'bot' : 'user'}`}>
              <div className="message-content">
                <p>{msg.text}</p>
                {msg.isBot && (
                  <div className="message-meta">
                    <span className="emotion-tag">{msg.emotion}</span>
                    <span className="source-tag">{msg.source}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="input-area">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !inputMessage.trim()}>
            {isLoading ? '...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;