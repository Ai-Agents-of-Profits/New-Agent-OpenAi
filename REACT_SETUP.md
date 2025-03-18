# Trading Assistant UI with React TypeScript and Tailwind CSS

This document provides instructions for running the Trading Agent with the modern React frontend.

## Project Structure

- `frontend/` - React TypeScript application with Tailwind CSS
- `web_app.py` - Flask backend that serves the API and the React frontend
- `deploy.py` - Utility script to build the React app for production

## Development Setup

### Running the Backend

```bash
# Start the Flask backend on port 8000
python web_app.py
```

### Running the Frontend in Development Mode

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start the development server
npm start
```

This will start the React development server on port 3000. The React app will proxy API requests to your Flask backend on port 8000.

## Production Deployment

To deploy the application for production:

1. Build the React frontend:

```bash
# Run the deploy script
python deploy.py
```

2. Start the Flask server:

```bash
# Start the server
python web_app.py
```

The Flask server will now serve both the API and the React frontend from a single origin (port 8000).

## Features

- **Minimalist UI** focused solely on agent interaction
- **Real-time communication** using WebSockets
- **Responsive design** with Tailwind CSS
- **TypeScript** for better type safety and developer experience
- **Modern React patterns** with hooks and functional components

## Usage Tips

- Enter your trading-related questions or commands in the input field
- The agent will respond in real-time through the WebSocket connection
- If WebSockets are unavailable, the app will automatically fall back to HTTP requests
