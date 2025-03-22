import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import { strategyPrompts, strategyDescriptions } from './data/tradingStrategies';

// Custom CSS classes for styling based on the provided theme preferences
const customStyles = {
  gradientBg: 'bg-gradient-to-br from-gray-900 to-gray-800',
  accentColor: 'text-purple-500',
  accentBg: 'bg-purple-600',
  accentHover: 'hover:bg-purple-700',
  buttonGradient: 'bg-gradient-to-r from-purple-600 to-pink-500',
  buttonHoverGradient: 'hover:from-purple-700 hover:to-pink-600',
  cardBg: 'bg-gray-800',
  borderColor: 'border-purple-500/30',
  textPrimary: 'text-white',
  textSecondary: 'text-gray-300',
  inputBg: 'bg-gray-700',
  fontDisplay: 'font-display',
  fontBody: 'font-body',
  sidebarGradient: 'bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900',
  cardGradient: 'bg-gradient-to-br from-gray-800 to-gray-900',
  agentCardGlow: 'bg-gradient-to-r from-purple-900/10 via-purple-800/5 to-transparent',
  activeAgentGlow: 'from-purple-900/20 via-purple-800/10 to-transparent',
  menuItemHover: 'hover:bg-white/5',
  inputFocus: 'focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50',
  buttonShadow: 'shadow-lg shadow-purple-500/20',
};

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
      text: 'Welcome to the Trading Agent! I coordinate specialized agents for market analysis, technical analysis, orderbook analysis, token dashboards, and trade execution. How can I assist you today?',
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

  // Load saved messages from localStorage on initial load
  useEffect(() => {
    const savedMessages = localStorage.getItem('trading-assistant-messages');
    if (savedMessages) {
      try {
        // Parse the saved messages and convert timestamps back to Date objects
        const parsedMessages: Message[] = JSON.parse(savedMessages).map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        if (parsedMessages.length > 0) {
          setMessages(parsedMessages);
        }
      } catch (e) {
        console.error('Error loading saved messages:', e);
      }
    }
  }, []);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 1) { // Only save if there are messages beyond the welcome message
      localStorage.setItem('trading-assistant-messages', JSON.stringify(messages));
    }
  }, [messages]);

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
      } else if (data.type === 'system') {
        // Handle system messages like conversation reset confirmation
        setIsProcessing(false);
        setMessages(prev => [...prev, {
          id: Date.now().toString(),
          text: data.data,
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

  // Check if the input is a reset command
  const isResetCommand = (text: string): boolean => {
    const lowerText = text.toLowerCase().trim();
    return lowerText === 'reset' || lowerText === 'new topic' || lowerText === 'clear history';
  };

  const handleSend = () => {
    if (input.trim() === '' || isProcessing) return;
    
    // Check if the user is requesting to reset the conversation
    if (isResetCommand(input)) {
      // Add the reset command as a user message
      const newMessage: Message = {
        id: Date.now().toString(),
        text: input,
        sender: 'user',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, newMessage]);
      setInput('');
      
      // Send the reset command to the server
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ message: input }));
      }
      
      setIsProcessing(true);
      return;
    }
    
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
    // Also clear localStorage
    localStorage.removeItem('trading-assistant-messages');
    
    // Send a reset command to the server to clear conversation history on the backend
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ message: "reset" }));
    }
  };

  return (
    <div className={`flex h-screen ${customStyles.gradientBg} ${customStyles.fontBody}`}>
      {/* Mobile Sidebar Overlay */}
      {isMobileSidebarOpen && (
        <div 
          className="md:hidden fixed inset-0 bg-gray-900 bg-opacity-70 z-20 backdrop-blur-sm"
          onClick={() => setIsMobileSidebarOpen(false)}
        ></div>
      )}
      
      {/* Sidebar */}
      <div className={`${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 fixed md:sticky top-0 left-0 h-full w-72 transition-transform duration-300 ease-in-out z-30 md:z-0 flex flex-col ${customStyles.sidebarGradient} border-r ${customStyles.borderColor} ${customStyles.textPrimary}`}>
        {/* Sidebar Header */}
        <div className={`px-6 py-6 border-b ${customStyles.borderColor} bg-gradient-to-r from-purple-900/20 to-transparent`}>
          <h1 className={`text-xl ${customStyles.fontDisplay} font-bold flex items-center bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent`}>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Agent of Profits
          </h1>
          <p className="text-sm mt-1 text-gray-400">Powered by Open AI Agent</p>
        </div>
        
        {/* Sidebar Content - Make this section scrollable */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6">
            <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-4">Available Agents</h2>
            <div className="mt-3 mb-6">
              <div className={`flex items-center text-white px-4 py-3 mb-2 rounded-md bg-gradient-to-r ${customStyles.activeAgentGlow} border-l-4 ${customStyles.borderColor}`}>
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center mr-3 shadow-lg shadow-purple-500/20">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <span className="font-semibold block">Orchestration Agent</span>
                  <span className="text-xs text-gray-400">Coordinates all specialized agents</span>
                </div>
              </div>
            </div>
            
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Specialized Agents:</h3>
            <div className="space-y-2">
              <div className={`flex items-center px-3 py-3 rounded-lg transition-all duration-200 ${customStyles.menuItemHover} ${customStyles.agentCardGlow}`}>
                <div className="w-8 h-8 rounded-full border border-purple-500/30 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
                <span className="text-sm">Execution Agent</span>
              </div>
              <div className={`flex items-center px-3 py-3 rounded-lg transition-all duration-200 ${customStyles.menuItemHover} ${customStyles.agentCardGlow}`}>
                <div className="w-8 h-8 rounded-full border border-purple-500/30 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                </div>
                <span className="text-sm">Technical Analysis Agent</span>
              </div>
              <div className={`flex items-center px-3 py-3 rounded-lg transition-all duration-200 ${customStyles.menuItemHover} ${customStyles.agentCardGlow}`}>
                <div className="w-8 h-8 rounded-full border border-purple-500/30 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0h2a2 2 0 012 2v2a2 2 0 01-2 2H9a2 2 0 01-2-2v-2a2 2 0 012-2z" />
                  </svg>
                </div>
                <span className="text-sm">Market Data Agent</span>
              </div>
              <div className={`flex items-center px-3 py-3 rounded-lg transition-all duration-200 ${customStyles.menuItemHover} ${customStyles.agentCardGlow}`}>
                <div className="w-8 h-8 rounded-full border border-purple-500/30 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
                <span className="text-sm">Orderbook Analysis Agent</span>
              </div>
              <div className={`flex items-center px-3 py-3 rounded-lg transition-all duration-200 ${customStyles.menuItemHover} ${customStyles.agentCardGlow}`}>
                <div className="w-8 h-8 rounded-full border border-purple-500/30 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <span className="text-sm">Token Dashboard Agent</span>
              </div>
            </div>
          </div>

          {/* Trading Strategies Section */}
          <div className="p-6 mt-2">
            <h3 className={`text-xs font-semibold uppercase tracking-wider ${customStyles.textSecondary} mb-3`}>Trading Strategies</h3>
            <div className="space-y-2">
              {Object.entries(strategyPrompts).map(([key, prompt]) => (
                <button
                  key={key}
                  onClick={() => handleStrategyClick(prompt)}
                  className={`w-full text-left px-3 py-3 rounded-lg ${
                    darkMode 
                      ? 'hover:bg-white/5 text-gray-300 border border-transparent hover:border-purple-500/20 transition-all duration-200' 
                      : 'hover:bg-gray-100 text-gray-700'
                  } text-sm flex items-center justify-between group`}
                >
                  <span>{strategyDescriptions[key as keyof typeof strategyDescriptions]}</span>
                  <svg xmlns="http://www.w3.org/2000/svg" className={`h-4 w-4 ${
                    darkMode 
                      ? 'text-purple-500/50 group-hover:text-purple-500' 
                      : 'text-gray-400 group-hover:text-gray-600'
                  } transition-colors duration-150`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              ))}
            </div>
          </div>
          
          {/* Resources */}
          <div className="p-6 border-t ${customStyles.borderColor}">
            <h3 className={`text-xs font-semibold uppercase tracking-wider ${customStyles.textSecondary} mb-3`}>Resources</h3>
            <div className="space-y-1">
              <a href="https://docs.example.com/trading-api" target="_blank" rel="noreferrer" className={`flex items-center px-3 py-2.5 text-sm ${customStyles.textPrimary} rounded-md ${customStyles.menuItemHover} transition-colors duration-200`}>
                <div className="w-7 h-7 rounded-full border border-purple-500/30 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                API Documentation
              </a>
              <a href="https://example.com/tutorials" target="_blank" rel="noreferrer" className={`flex items-center px-3 py-2.5 text-sm ${customStyles.textPrimary} rounded-md ${customStyles.menuItemHover} transition-colors duration-200`}>
                <div className="w-7 h-7 rounded-full border border-purple-500/30 flex items-center justify-center mr-3">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </div>
                Tutorials
              </a>
            </div>
          </div>

          {/* Theme switcher button in the sidebar footer */}
          <div className={`mt-auto p-6 border-t ${customStyles.borderColor} bg-gradient-to-r from-purple-900/10 to-transparent flex justify-between items-center`}>
            <div className="flex items-center text-sm text-gray-400">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2 shadow-lg shadow-green-500/50 animate-pulse"></div>
              <span>System Online</span>
            </div>
            
            {/* Theme toggle button */}
            <button 
              onClick={toggleTheme}
              className={`p-2 rounded-full ${
                darkMode 
                  ? 'bg-gradient-to-br from-gray-700 to-gray-800 text-purple-400 hover:text-purple-300' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              } transition-all duration-200 shadow-md`}
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
        <header className={`py-4 px-6 flex items-center justify-between shadow-lg z-10 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 border-b ${customStyles.borderColor} ${customStyles.textPrimary}`}>
          <div className="flex-1 min-w-0 flex items-center">
            <button 
              className="md:hidden p-2 mr-3 rounded-md bg-gray-800/50 hover:bg-gray-700 text-purple-400 transition-all duration-200 border border-purple-500/20"
              onClick={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="flex items-center">
              <div className="w-8 h-8 mr-2 rounded-md bg-gradient-to-br from-purple-600 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h2 className={`text-xl ${customStyles.fontDisplay} font-bold bg-gradient-to-r from-white via-purple-100 to-white bg-clip-text text-transparent`}>Trading Agent</h2>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="hidden md:flex items-center mr-2 text-xs text-gray-400">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse shadow-sm shadow-green-500/50"></div>
              <span>Connected</span>
            </div>
            <button 
              onClick={handleClearChat} 
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                darkMode 
                  ? 'bg-gradient-to-r from-gray-700 to-gray-800 text-gray-200 hover:from-gray-600 hover:to-gray-700 border border-purple-500/20 shadow-md' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200 border border-gray-200'
              }`}
            >
              <span className="flex items-center">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Clear Chat
              </span>
            </button>
          </div>
        </header>
        
        {/* Messages */}
        <div className={`flex-1 overflow-hidden p-6 chat-container ${customStyles.cardGradient} ${customStyles.textPrimary}`}>
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
                        ? `${customStyles.accentBg} ${customStyles.textPrimary} ml-auto mr-4 max-w-2xl user-message` 
                        : `${customStyles.cardGradient} ${customStyles.borderColor} ${customStyles.textPrimary} border ml-4 mr-auto max-w-2xl agent-message`
                    }`}
                    style={{
                      transform: `translateY(${index * 2}px) rotate(${message.sender === 'user' ? '-0.5' : '0.5'}deg)`,
                      zIndex: messages.length - index
                    }}
                  >
                    {message.sender === 'agent' && (
                      <div className={`flex items-center p-3 border-b ${customStyles.borderColor}`}>
                        <div className={`w-8 h-8 rounded-full ${customStyles.accentBg} flex items-center justify-center mr-2`}>
                          <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 ${customStyles.accentColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                          </svg>
                        </div>
                        <span className={`font-medium ${customStyles.textPrimary}`}>Orchestration Agent</span>
                      </div>
                    )}
                    <div className={`p-4 ${message.sender === 'user' ? customStyles.textPrimary : customStyles.textPrimary}`}>
                      {message.sender === 'agent' ? formatMessage(message.text) : message.text}
                    </div>
                    <div className={`px-4 pb-2 text-xs ${
                      message.sender === 'user' ? 'text-blue-200' : customStyles.textSecondary
                    }`}>
                      {message.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                    </div>
                  </div>
                </div>
              ))}
              {isProcessing && (
                <div className="message-container">
                  <div className={`${customStyles.cardGradient} ${customStyles.borderColor} border rounded-lg p-4 shadow ml-4 mr-auto max-w-2xl`} style={{ transform: 'rotate(0.5deg)' }}>
                    <div className="flex items-center">
                      <div className="w-8 h-8 rounded-full border border-purple-500/30 flex items-center justify-center mr-3 bg-gradient-to-br from-purple-600/20 to-pink-600/10">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                      </div>
                      <div className="flex flex-col">
                        <span className={`${customStyles.fontDisplay} text-sm font-semibold bg-gradient-to-r from-purple-400 to-pink-300 bg-clip-text text-transparent`}>Agent is thinking...</span>
                        <div className="typing-indicator mt-1">
                          <span className="bg-purple-400"></span>
                          <span className="bg-pink-400"></span>
                          <span className="bg-purple-300"></span>
                        </div>
                      </div>
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
          <div className="px-6 py-3 bg-transparent">
            <p className={`text-xs ${customStyles.textSecondary} mb-2`}>Try asking:</p>
            <div className="flex flex-wrap gap-2">
              {quickSuggestions.map((suggestion, index) => (
                <button 
                  key={index}
                  onClick={() => {
                    setInput(suggestion);
                    setTimeout(() => handleSend(), 100);
                  }}
                  className={`px-3 py-1.5 ${customStyles.textPrimary} text-sm rounded-full bg-white/5 hover:bg-white/10 border border-purple-500/30 transition-colors duration-200`}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
        
        {/* Input */}
        <div className={`p-4 border-t ${customStyles.borderColor}`}>
          <div className={`flex items-start ${customStyles.inputBg} rounded-lg p-1 shadow-inner transition-all duration-300 focus-within:shadow-purple-500/20 focus-within:border focus-within:border-purple-500/30`}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSend()}
              placeholder="Ask about market data, technical analysis, trade execution, or any crypto trading topic... (Shift+Enter for new line)"
              className={`flex-1 bg-transparent px-3 py-2 rounded ${customStyles.inputFocus} focus:outline-none resize-none min-h-[44px] max-h-[120px] overflow-y-auto ${customStyles.textPrimary}`}
              disabled={isProcessing}
              rows={Math.min(4, input.split('\n').length || 1)}
              style={{ height: `${Math.min(120, Math.max(44, 20 * (input.split('\n').length || 1)))}px` }}
            />
            <button
              onClick={handleSend}
              disabled={isProcessing || !input.trim()}
              className={`p-2 rounded-md text-white self-end transition-all duration-300 transform ${
                isProcessing || !input.trim() 
                  ? 'bg-gray-500/50 cursor-not-allowed' 
                  : `${customStyles.buttonGradient} ${customStyles.buttonShadow} hover:scale-105 ${customStyles.buttonHoverGradient}`
              }`}
              aria-label="Send message"
            >
              {isProcessing ? (
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m-7-7H3" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
