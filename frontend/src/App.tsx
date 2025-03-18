import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { strategyPrompts, strategyDescriptions } from './data/tradingStrategies';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'agent';
  timestamp: Date;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      text: 'Welcome to the Trading Assistant! I coordinate specialized agents for market analysis, technical analysis, orderbook analysis, token dashboards, and trade execution. How can I assist you today?',
      sender: 'agent',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(true); // Default to dark mode
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Connect to WebSocket
    ws.current = new WebSocket(`ws://127.0.0.1:8000/ws`);
    
    ws.current.onopen = () => {
      console.log('WebSocket connected');
    };
    
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'processing') {
        setIsProcessing(true);
      } else if (data.type === 'response') {
        setIsProcessing(false);
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          text: data.data.text,
          sender: 'agent',
          timestamp: new Date(data.data.timestamp * 1000),
        }]);
      } else if (data.type === 'error') {
        setIsProcessing(false);
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          text: `Error: ${data.data}`,
          sender: 'agent',
          timestamp: new Date(),
        }]);
      }
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsProcessing(false);
    };
    
    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
    };
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  useEffect(() => {
    // Scroll to bottom on new messages
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Check localStorage for saved theme preference
    const savedTheme = localStorage.getItem('profitmus-theme');
    
    // Apply dark theme if saved or if user's system preference is dark
    if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      setDarkMode(true);
    } else {
      setDarkMode(false);
    }
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-theme');
    } else {
      document.body.classList.remove('dark-theme');
    }
    
    // Save theme preference to localStorage
    localStorage.setItem('profitmus-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const toggleTheme = () => {
    setDarkMode(!darkMode);
  };

  const handleSend = () => {
    if (input.trim() === '' || isProcessing) return;
    
    const newMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, newMessage]);
    setInput('');
    
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ message: input }));
    } else {
      // Fallback to HTTP if WebSocket is not available
      fetch('/send_message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      })
      .then(response => response.json())
      .then(data => {
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          text: data.response,
          sender: 'agent',
          timestamp: new Date(),
        }]);
        setIsProcessing(false);
      })
      .catch(error => {
        console.error('Error:', error);
        setIsProcessing(false);
      });
    }
    
    setIsProcessing(true);
  };

  const quickSuggestions = [
    "What's the current price of BTC/USDT?",
    "Analyze the orderbook for ETH/USDT",
    "Perform technical analysis on SOL/USDT"
  ];

  // Function to format message text with markdown-like features
  const formatMessage = (text: string) => {
    // Handle snippet identifiers first
    const formattedText = text
      // Format snippet identifiers
      .replace(/\[Snippet identifier=([^\]]+)\]/g, '<div class="snippet-identifier"><span class="snippet-label">Snippet:</span> <code>$1</code></div>')
      // Format code blocks with language highlighting
      .replace(/```([a-z]*)\n([\s\S]*?)```/g, (match, language, code) => {
        return `<div class="code-block"><div class="code-header">${language || 'code'}</div><pre class="${darkMode ? 'bg-gray-900' : 'bg-gray-100'} ${darkMode ? 'text-gray-100' : 'text-gray-800'} p-3 rounded my-2 overflow-auto"><code>${code.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre></div>`;
      })
      // Format inline code
      .replace(/`([^`]+)`/g, `<code class="${darkMode ? 'bg-gray-700 text-blue-300' : 'bg-gray-200 text-blue-700'} px-1 py-0.5 rounded">$1</code>`)
      // Format headings (## Heading)
      .replace(/^##\s+(.+)$/gm, '<h2 class="text-xl font-bold mt-4 mb-2 border-b pb-1 border-gray-300">$1</h2>')
      // Format subheadings (### Subheading)
      .replace(/^###\s+(.+)$/gm, '<h3 class="text-lg font-semibold mt-3 mb-1">$1</h3>')
      // Format bold text
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      // Format italic text
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      // Add proper newlines with spacing
      .replace(/\n\n/g, '<br/><br/>');
    
    return <div dangerouslySetInnerHTML={{ __html: formattedText }} />;
  };

  const handleStrategyClick = (prompt: string) => {
    setInput(prompt);
    if (window.innerWidth < 768) {
      setIsMobileSidebarOpen(false);
    }
  };
  
  const handleClearChat = () => {
    // Keep only the first message (welcome message)
    setMessages(messages.length > 0 ? [messages[0]] : []);
  };

  return (
    <div className="flex h-screen bg-gradient-to-b from-gray-50 to-gray-100">
      {/* Mobile Sidebar Overlay */}
      {isMobileSidebarOpen && (
        <div 
          className="md:hidden fixed inset-0 bg-gray-600 bg-opacity-50 z-20"
          onClick={() => setIsMobileSidebarOpen(false)}
        ></div>
      )}
      
      {/* Sidebar */}
      <div className={`${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 fixed md:sticky top-0 left-0 h-full w-72 transition-transform duration-300 ease-in-out z-30 md:z-0 flex flex-col ${darkMode ? 'bg-gray-900 border-gray-700 text-gray-100' : 'bg-white border-gray-200'} border-r`}>
        {/* Sidebar Header */}
        <div className={`px-6 py-6 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
          <h1 className={`text-xl font-bold flex items-center ${darkMode ? 'text-blue-400' : 'text-blue-600'}`}>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Profitmus
          </h1>
          <p className={`text-sm mt-1 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>Powered by Open AI Agent</p>
        </div>
        
        {/* Sidebar Content - Make this section scrollable */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Available Agents</h2>
            <div className="mt-3 mb-4">
              <div className="flex items-center text-gray-700 px-3 py-2 mb-2 rounded-md bg-blue-50 border-l-4 border-blue-500">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span className="font-semibold">Orchestration Agent</span>
              </div>
              <p className="text-xs text-gray-500 px-3">Coordinates all specialized agents based on your questions</p>
            </div>
            
            <h3 className="text-xs font-semibold text-gray-500 px-3 mb-2">Specialized Agents:</h3>
            <div className="space-y-2">
              <div className="flex items-center text-gray-700 px-3 py-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span>Execution Agent</span>
              </div>
              <div className="flex items-center text-gray-700 px-3 py-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
                <span>Technical Analysis Agent</span>
              </div>
              <div className="flex items-center text-gray-700 px-3 py-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Market Data Agent</span>
              </div>
              <div className="flex items-center text-gray-700 px-3 py-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>Orderbook Analysis Agent</span>
              </div>
              <div className="flex items-center text-gray-700 px-3 py-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>Token Dashboard Agent</span>
              </div>
            </div>
          </div>

          {/* Trading Strategies Section */}
          <div className="p-6">
            <h3 className={`text-sm font-medium uppercase tracking-wider ${darkMode ? 'text-gray-400' : 'text-gray-500'} mb-3`}>Trading Strategies</h3>
            <div className="space-y-2">
              {Object.entries(strategyPrompts).map(([key, prompt]) => (
                <button
                  key={key}
                  onClick={() => handleStrategyClick(prompt)}
                  className={`w-full text-left px-3 py-2 rounded-lg ${
                    darkMode 
                      ? 'hover:bg-gray-800 text-gray-300' 
                      : 'hover:bg-gray-100 text-gray-700'
                  } text-sm transition-colors duration-150 flex items-center justify-between group`}
                >
                  <span>{strategyDescriptions[key as keyof typeof strategyDescriptions]}</span>
                  <svg xmlns="http://www.w3.org/2000/svg" className={`h-4 w-4 ${
                    darkMode 
                      ? 'text-gray-500 group-hover:text-gray-400' 
                      : 'text-gray-400 group-hover:text-gray-600'
                  } transition-colors duration-150`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              ))}
            </div>
          </div>
          
          {/* Resources */}
          <div className="p-6 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}">
            <h3 className={`text-sm font-medium uppercase tracking-wider ${darkMode ? 'text-gray-400' : 'text-gray-500'} mb-3`}>Resources</h3>
            <a href="https://docs.example.com/trading-api" target="_blank" rel="noreferrer" className={`flex items-center text-gray-700 hover:text-blue-600 px-3 py-2 text-sm ${darkMode ? 'text-gray-300' : ''}`}>
              <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 mr-2 text-gray-400 ${darkMode ? 'text-gray-500' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              API Documentation
            </a>
            <a href="https://example.com/tutorials" target="_blank" rel="noreferrer" className={`flex items-center text-gray-700 hover:text-blue-600 px-3 py-2 text-sm ${darkMode ? 'text-gray-300' : ''}`}>
              <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 mr-2 text-gray-400 ${darkMode ? 'text-gray-500' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Tutorials
            </a>
          </div>

          {/* Theme switcher button in the sidebar footer */}
          <div className={`mt-auto p-6 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'} flex justify-between items-center`}>
            <div className="flex items-center text-sm text-gray-500">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
              <span>System Online</span>
            </div>
            
            {/* Theme toggle button */}
            <button 
              onClick={toggleTheme}
              className={`p-2 rounded-full ${
                darkMode 
                  ? 'bg-gray-700 text-blue-400 hover:bg-gray-600' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              } transition-colors`}
              aria-label="Toggle theme"
            >
              {darkMode ? (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className={`py-4 px-6 flex items-center justify-between shadow-sm z-10 ${darkMode ? 'bg-gray-800 shadow-gray-900' : 'bg-white shadow-gray-100'}`}>
          <div className="flex-1 min-w-0 flex items-center">
            <button 
              className="md:hidden p-2 mr-3 rounded-md text-gray-500 hover:text-gray-600 hover:bg-gray-100 focus:outline-none"
              onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <h2 className={`text-xl font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>Trading Assistant</h2>
          </div>
          
          <div className="flex space-x-4">
            <button 
              onClick={handleClearChat} 
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                darkMode 
                  ? 'bg-gray-700 text-gray-200 hover:bg-gray-600' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Clear Chat
            </button>
          </div>
        </header>
        
        {/* Messages */}
        <div className={`flex-1 overflow-hidden p-6 chat-container ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
          <div className="h-full overflow-y-auto pr-4">
            <div className="space-y-4">
              {messages.map((message, index) => (
                <div 
                  key={message.id} 
                  className="message-container relative"
                  style={{
                    marginTop: index > 0 ? '-10px' : '0'
                  }}
                >
                  <div 
                    className={`rounded-lg shadow message-bubble ${
                      message.sender === 'user' 
                        ? `${darkMode ? 'bg-blue-800' : 'bg-blue-600'} text-white ml-auto mr-4 max-w-2xl user-message` 
                        : `${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} ${darkMode ? 'text-gray-200' : ''} border ml-4 mr-auto max-w-2xl agent-message`
                    }`}
                    style={{
                      transform: `translateY(${index * 2}px) rotate(${message.sender === 'user' ? '-0.5' : '0.5'}deg)`,
                      zIndex: messages.length - index
                    }}
                  >
                    {message.sender === 'agent' && (
                      <div className={`flex items-center p-3 border-b ${darkMode ? 'border-gray-600' : 'border-gray-100'}`}>
                        <div className={`w-8 h-8 rounded-full ${darkMode ? 'bg-blue-900' : 'bg-blue-100'} flex items-center justify-center mr-2`}>
                          <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                          </svg>
                        </div>
                        <span className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>Orchestration Agent</span>
                      </div>
                    )}
                    <div className={`p-4 ${message.sender === 'user' ? 'text-white' : darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                      {message.sender === 'agent' ? formatMessage(message.text) : message.text}
                    </div>
                    <div className={`px-4 pb-2 text-xs ${
                      message.sender === 'user' ? 'text-blue-200' : darkMode ? 'text-gray-400' : 'text-gray-500'
                    }`}>
                      {message.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                    </div>
                  </div>
                </div>
              ))}
              {isProcessing && (
                <div className="message-container">
                  <div className={`${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border rounded-lg p-4 shadow ml-4 mr-auto max-w-2xl`} style={{ transform: 'rotate(0.5deg)' }}>
                    <div className="flex items-center">
                      <div className="typing-indicator mr-2">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                      <span className={`${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>Processing your request...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>
        </div>
        
        {/* Quick Suggestions */}
        {!isProcessing && messages.length < 3 && (
          <div className="px-6 py-3 bg-white border-t border-gray-200">
            <p className="text-xs text-gray-500 mb-2">Try asking:</p>
            <div className="flex flex-wrap gap-2">
              {quickSuggestions.map((suggestion, index) => (
                <button 
                  key={index}
                  onClick={() => {
                    setInput(suggestion);
                    setTimeout(() => handleSend(), 100);
                  }}
                  className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-800 text-sm rounded-full"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
        
        {/* Input */}
        <div className={`p-4 border-t ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
          <div className={`flex items-start ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} rounded-lg p-1 shadow-inner`}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Ask about market data, technical analysis, trade execution, or any crypto trading topic... (Shift+Enter for new line)"
              className={`flex-1 bg-transparent px-3 py-2 rounded focus:outline-none resize-none min-h-[44px] max-h-[120px] overflow-y-auto ${darkMode ? 'text-gray-200 placeholder-gray-400' : ''}`}
              disabled={isProcessing}
              rows={Math.min(4, input.split('\n').length || 1)}
              style={{ height: `${Math.min(120, Math.max(44, 20 * (input.split('\n').length || 1)))}px` }}
            />
            <button
              onClick={handleSend}
              disabled={isProcessing || !input.trim()}
              className={`p-2 rounded-md text-white self-end ${
                isProcessing || !input.trim() ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m-7-7H3" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
