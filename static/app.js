// Crypto Trading Assistant Frontend JavaScript

// Web Socket Connection
let socket = null;
let isConnected = false;

// DOM Elements
const messageInput = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const messagesContainer = document.getElementById('chat-messages');
const connectionStatus = document.querySelector('.status-text');
const statusDot = document.querySelector('.status-dot');

// Initialize the connection
function initializeWebSocket() {
    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    // Connection opened
    socket.addEventListener('open', (event) => {
        isConnected = true;
        updateConnectionStatus(true);
        console.log('Connected to the server');
    });
    
    // Listen for messages
    socket.addEventListener('message', (event) => {
        const message = JSON.parse(event.data);
        console.log('Message from server:', message);
        
        if (message.type === 'response') {
            displayMessage(message.data.text, 'assistant');
        } else if (message.type === 'processing') {
            // Add typing indicator or processing message
            addProcessingIndicator();
        } else if (message.type === 'error') {
            addErrorMessage(message.data);
        }
    });
    
    // Connection closed
    socket.addEventListener('close', (event) => {
        isConnected = false;
        updateConnectionStatus(false);
        console.log('Disconnected from the server');
        
        // Try to reconnect after 5 seconds
        setTimeout(initializeWebSocket, 5000);
    });
    
    // Connection error
    socket.addEventListener('error', (error) => {
        isConnected = false;
        updateConnectionStatus(false);
        console.error('WebSocket error:', error);
    });
}

// Update the connection status indicator
function updateConnectionStatus(connected) {
    if (connected) {
        connectionStatus.textContent = 'Connected';
        statusDot.classList.add('connected');
        statusDot.classList.remove('disconnected');
    } else {
        connectionStatus.textContent = 'Disconnected';
        statusDot.classList.add('disconnected');
        statusDot.classList.remove('connected');
    }
}

// Function to format trading analysis content with specialized styling
function formatTradingAnalysis(content) {
    // Check if the content appears to be a trading analysis
    if (content.includes("Analysis for") && (
        content.includes("Setup") || 
        content.includes("Entry") || 
        content.includes("Exit") || 
        content.includes("Support") || 
        content.includes("Resistance")
    )) {
        // Extract the symbol and timeframe from the header
        const headerMatch = content.match(/Analysis for ([A-Z0-9]+\/[A-Z0-9]+)/);
        const symbol = headerMatch ? headerMatch[1] : "Trading Pair";
        
        // Find timeframe if available
        const timeframeMatch = content.match(/on the (\w+) timeframe/);
        const timeframe = timeframeMatch ? timeframeMatch[1].toUpperCase() : "Multiple Timeframes";
        
        // Start building the HTML structure
        let formattedHtml = `
            <div class="trading-analysis">
                <div class="trading-analysis-header">
                    <span class="symbol">${symbol} Analysis</span>
                    <span class="timeframe">${timeframe}</span>
                </div>
                <div class="trading-analysis-content">`;
        
        // Process the content by sections
        const sections = content.split(/####\s+\d+\.\s+\*\*([^*]+)\*\*/g);
        
        if (sections.length > 1) {
            // Process structured content with sections
            for (let i = 1; i < sections.length; i += 2) {
                const sectionTitle = sections[i];
                const sectionContent = sections[i + 1];
                
                // Add appropriate icon based on section title
                let icon = "fas fa-chart-line";
                if (sectionTitle.toLowerCase().includes("order book")) {
                    icon = "fas fa-book";
                } else if (sectionTitle.toLowerCase().includes("technical")) {
                    icon = "fas fa-chart-bar";
                } else if (sectionTitle.toLowerCase().includes("volume")) {
                    icon = "fas fa-chart-area";
                } else if (sectionTitle.toLowerCase().includes("bid-ask")) {
                    icon = "fas fa-exchange-alt";
                } else if (sectionTitle.toLowerCase().includes("setup")) {
                    icon = "fas fa-bolt";
                }
                
                formattedHtml += `
                    <div class="trading-analysis-section">
                        <div class="trading-analysis-section-title">
                            <i class="${icon}"></i> ${sectionTitle}
                        </div>`;
                
                // Check if this is the setups section
                if (sectionTitle.toLowerCase().includes("setup") || sectionTitle.toLowerCase().includes("trade")) {
                    // Process potential trade setups
                    const setups = sectionContent.split(/#{5}\s+Setup \d+:|#{5}\s+[^#]+/g).filter(s => s.trim());
                    
                    if (setups.length > 0) {
                        for (const setup of setups) {
                            if (setup.trim()) {
                                const setupMatch = setup.match(/([^:]+):/);
                                const setupTitle = setupMatch ? setupMatch[1].trim() : "Trading Setup";
                                
                                formattedHtml += `<div class="trading-setup">
                                    <div class="trading-setup-title">${setupTitle}</div>`;
                                
                                // Extract entry, exit, risk management info
                                const entryMatch = setup.match(/Entry:([^<\n]+)/);
                                const exitMatch = setup.match(/Exit:([^<\n]+)/);
                                const riskMatch = setup.match(/Risk Management:([^<\n]+)/) || setup.match(/Invalidation([^<\n]+)/);
                                const durationMatch = setup.match(/Duration:([^<\n]+)/);
                                const rrMatch = setup.match(/R\/R Ratio:([^<\n]+)/);
                                
                                formattedHtml += `<div class="trading-setup-details">`;
                                
                                // Entry
                                if (entryMatch) {
                                    const entryText = entryMatch[1].trim();
                                    let entryClass = "neutral";
                                    if (entryText.toLowerCase().includes("buy") || entryText.toLowerCase().includes("long")) {
                                        entryClass = "bullish";
                                    } else if (entryText.toLowerCase().includes("sell") || entryText.toLowerCase().includes("short")) {
                                        entryClass = "bearish";
                                    }
                                    
                                    formattedHtml += `
                                        <div class="setup-detail">
                                            <span class="setup-detail-label">ENTRY</span>
                                            <span class="setup-detail-value ${entryClass}">${entryText}</span>
                                        </div>`;
                                }
                                
                                // Exit
                                if (exitMatch) {
                                    formattedHtml += `
                                        <div class="setup-detail">
                                            <span class="setup-detail-label">EXIT</span>
                                            <span class="setup-detail-value">${exitMatch[1].trim()}</span>
                                        </div>`;
                                }
                                
                                // Risk Management
                                if (riskMatch) {
                                    formattedHtml += `
                                        <div class="setup-detail">
                                            <span class="setup-detail-label">RISK</span>
                                            <span class="setup-detail-value">${riskMatch[1].trim()}</span>
                                        </div>`;
                                }
                                
                                // Duration
                                if (durationMatch) {
                                    formattedHtml += `
                                        <div class="setup-detail">
                                            <span class="setup-detail-label">DURATION</span>
                                            <span class="setup-detail-value">${durationMatch[1].trim()}</span>
                                        </div>`;
                                }
                                
                                // R/R Ratio
                                if (rrMatch) {
                                    const rrValue = rrMatch[1].trim();
                                    let rrClass = "moderate";
                                    
                                    // Parse the R/R ratio to determine if it's good
                                    const rrNumber = parseFloat(rrValue);
                                    if (!isNaN(rrNumber)) {
                                        if (rrNumber >= 2) {
                                            rrClass = "good";
                                        } else if (rrNumber < 1) {
                                            rrClass = "poor";
                                        }
                                    }
                                    
                                    formattedHtml += `
                                        <div class="setup-detail">
                                            <span class="setup-detail-label">RISK/REWARD</span>
                                            <span class="risk-reward ${rrClass}">${rrValue}</span>
                                        </div>`;
                                }
                                
                                formattedHtml += `</div></div>`;
                            }
                        }
                    } else {
                        // If no specific setups found, just render the content
                        formattedHtml += formatSectionContent(sectionContent);
                    }
                } else {
                    // Regular section content
                    formattedHtml += formatSectionContent(sectionContent);
                }
                
                formattedHtml += `</div>`;
            }
        } else {
            // Less structured content, format it as best we can
            formattedHtml += `<div class="trading-analysis-section">${markdownToHtml(content)}</div>`;
        }
        
        formattedHtml += `</div></div>`;
        return formattedHtml;
    }
    
    // Not a trading analysis, use standard markdown formatting
    return markdownToHtml(content);
}

// Helper function to format section content in trading analysis
function formatSectionContent(content) {
    // Check if we have bullet points
    if (content.includes("-")) {
        const lines = content.split('\n').filter(line => line.trim());
        let html = '<ul class="trading-analysis-list">';
        
        for (const line of lines) {
            const trimmedLine = line.trim();
            if (trimmedLine.startsWith('-')) {
                const listItemContent = trimmedLine.substring(1).trim();
                if (listItemContent) {
                    html += `<li>${listItemContent}</li>`;
                }
            } else if (trimmedLine) {
                // For non-list lines, render them normally
                html += `<p>${trimmedLine}</p>`;
            }
        }
        
        html += '</ul>';
        return html;
    } else {
        // Regular content
        return `<p>${content.trim()}</p>`;
    }
}

// Convert markdown to HTML
function markdownToHtml(markdown) {
    // Basic markdown to HTML conversion
    let html = markdown
        // Replace headers
        .replace(/^##### (.*$)/gim, '<h5>$1</h5>')
        .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
        .replace(/^### (.*$)/gim, '<h3>$1</h3>')
        .replace(/^## (.*$)/gim, '<h2>$1</h2>')
        .replace(/^# (.*$)/gim, '<h1>$1</h1>')
        
        // Replace bold and italic
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        
        // Replace lists
        .replace(/^\s*-\s*(.*?)$/gm, '<li>$1</li>')
        
        // Replace links
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        
        // Replace code blocks
        .replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>')
        
        // Replace inline code
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        
        // Handle paragraphs and line breaks
        .replace(/\n\s*\n/g, '</p><p>')
        .replace(/\n/g, '<br>');
    
    html = '<p>' + html + '</p>';
    
    // Wrap lists in ul tags
    if (html.includes('<li>')) {
        html = html.replace(/<p>(<li>.*?<\/li>)<\/p>/gs, '<ul>$1</ul>');
    }
    
    return html;
}

// Display message in chat
function displayMessage(message, sender) {
    const chatMessages = document.getElementById('chat-messages');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', sender);
    
    const contentElement = document.createElement('div');
    contentElement.classList.add('message-content');
    
    // Format message content based on type
    if (sender === 'assistant') {
        contentElement.innerHTML = formatTradingAnalysis(message);
    } else {
        contentElement.innerHTML = markdownToHtml(message);
    }
    
    messageElement.appendChild(contentElement);
    chatMessages.appendChild(messageElement);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add a message to the chat
function addMessage(content, sender) {
    // Remove any existing processing indicators
    const processingIndicators = document.querySelectorAll('.processing-indicator');
    processingIndicators.forEach(indicator => {
        indicator.remove();
    });
    
    displayMessage(content, sender);
}

// Add a processing indicator
function addProcessingIndicator() {
    // Check if there's already a processing indicator
    if (document.querySelector('.processing-indicator')) {
        return;
    }
    
    const processingElement = document.createElement('div');
    processingElement.classList.add('message', 'assistant', 'processing-indicator');
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    
    messageContent.innerHTML = `
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    processingElement.appendChild(messageContent);
    messagesContainer.appendChild(processingElement);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Add an error message
function addErrorMessage(errorText) {
    // Remove any existing processing indicators
    const processingIndicators = document.querySelectorAll('.processing-indicator');
    processingIndicators.forEach(indicator => {
        indicator.remove();
    });
    
    const errorElement = document.createElement('div');
    errorElement.classList.add('message', 'error');
    
    const messageContent = document.createElement('div');
    messageContent.classList.add('message-content');
    
    messageContent.innerHTML = `<p><i class="fas fa-exclamation-triangle"></i> ${errorText}</p>`;
    errorElement.appendChild(messageContent);
    
    messagesContainer.appendChild(errorElement);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Send a message
function sendMessage() {
    const message = messageInput.value.trim();
    
    if (message && isConnected) {
        // Add message to the chat
        addMessage(message, 'user');
        
        // Send message to the server
        socket.send(JSON.stringify({
            message: message
        }));
        
        // Clear input
        messageInput.value = '';
    }
}

// Event listeners
sendButton.addEventListener('click', sendMessage);

messageInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

// Initialize the WebSocket connection when the page loads
window.addEventListener('load', initializeWebSocket);

// Add click handlers for strategy and agent items
document.querySelectorAll('.strategies-list li, .agents-list li').forEach(item => {
    item.addEventListener('click', function() {
        const text = this.textContent.trim();
        messageInput.value = `Tell me more about the ${text}`;
        messageInput.focus();
    });
});
