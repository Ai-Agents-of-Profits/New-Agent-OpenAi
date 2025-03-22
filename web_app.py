"""
Web application for the enhanced crypto market analysis system.
Provides a modern UI for interacting with the agent.
"""
import asyncio
import logging
import os
import json
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from enhanced_main import setup_crypto_agents, handle_user_message, ConversationHistory
from agents import Runner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Crypto Trading Agent", description="Web interface for the enhanced crypto market analysis system")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversation_histories: Dict[WebSocket, List[Dict]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Initialize empty conversation history for new connections
        self.conversation_histories[websocket] = []

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        # Clean up conversation history when connection is closed
        if websocket in self.conversation_histories:
            del self.conversation_histories[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    def get_conversation_history(self, websocket: WebSocket):
        return self.conversation_histories.get(websocket, [])
    
    def add_to_history(self, websocket: WebSocket, role: str, content: str):
        if websocket not in self.conversation_histories:
            self.conversation_histories[websocket] = []
        self.conversation_histories[websocket].append({"role": role, "content": content})
    
    def reset_history(self, websocket: WebSocket):
        self.conversation_histories[websocket] = []

manager = ConnectionManager()

# Initialize agent on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the web application...")
    try:
        app.state.agent = await setup_crypto_agents()
        logger.info("Agents initialized successfully!")
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize agents: {str(e)}")

# Initialize conversation history storage for API endpoints
@app.on_event("startup")
async def initialize_api_conversations():
    app.state.api_conversation_histories = {}

# WebSocket endpoint for chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Import the ConversationHistory class
        from enhanced_main import ConversationHistory
        
        # Create a ConversationHistory object for this connection
        conversation = ConversationHistory()
        
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            user_message = data_json.get("message", "")
            if not user_message:
                continue
            
            # Check for reset command
            if user_message.lower() in ["reset", "new topic", "clear history"]:
                conversation.clear()
                manager.reset_history(websocket)
                await manager.send_personal_message(
                    json.dumps({
                        "type": "system",
                        "data": "Conversation history has been cleared. Starting a new conversation."
                    }),
                    websocket
                )
                continue
                
            # Send acknowledgment
            await manager.send_personal_message(
                json.dumps({"type": "processing", "data": "Processing your request..."}),
                websocket
            )
            
            # Add user message to history
            conversation.add_user_message(user_message)
            
            # Process the message with the agent
            try:
                logger.info(f"Processing request: {user_message}")
                
                # Run the agent with conversation history if available
                if len(conversation.messages) > 1:
                    result = await Runner.run(app.state.agent, conversation.get_input_list())
                else:
                    result = await Runner.run(app.state.agent, user_message)
                
                # Get the agent's response
                response = result.final_output
                
                # Add assistant response to history
                conversation.add_assistant_message(response)
                
                # Also update the connection manager history for backup
                manager.add_to_history(websocket, "assistant", response)
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "response",
                        "data": {
                            "text": response,
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    }),
                    websocket
                )
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                # Don't reset conversation history on every error - only do this for critical errors
                # Add the error to the conversation without clearing history
                error_message = f"I encountered an issue while processing your request. Please try again or modify your query."
                
                # Only add the error message to the conversation, don't clear it
                conversation.add_assistant_message(error_message)
                manager.add_to_history(websocket, "assistant", error_message)
                
                await manager.send_personal_message(
                    json.dumps({
                        "type": "response",
                        "data": {
                            "text": error_message,
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    }),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# API endpoint for sending a message (alternative to WebSocket)
@app.post("/send_message")
async def send_message(request: Request):
    """
    Endpoint to process a message from the user and send a response.
    """
    try:
        data = await request.json()
        user_message = data.get("message", "")
        session_id = data.get("session_id")
        
        # Get or create conversation history for this session
        conversation_history = None
        if session_id and hasattr(app.state, "api_conversation_histories"):
            conversation_history = app.state.api_conversation_histories.get(session_id)
        
        # Process the message using our enhanced_main module with conversation history
        response, updated_history = await handle_user_message(user_message, conversation_history)
        
        # Store the updated conversation history if we have a session
        if session_id and updated_history and hasattr(app.state, "api_conversation_histories"):
            app.state.api_conversation_histories[session_id] = updated_history
        
        return {"response": response, "session_id": session_id}
    except Exception as e:
        logger.exception(f"Error processing message: {e}")
        return {"response": f"I encountered an error: {str(e)}"}

# Check if running in development mode
DEV_MODE = os.environ.get('FLASK_ENV') == 'development'

if DEV_MODE:
    # In development, serve static files from the React development server
    @app.get("/")
    async def redirect_to_react():
        return JSONResponse({"message": "API server running. Frontend is at http://localhost:3000"})
else:
    # In production, serve the React build
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "frontend", "build")
    if os.path.exists(STATIC_DIR):
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    else:
        # Fallback to the old static directory if React build doesn't exist
        app.mount("/static", StaticFiles(directory="static"), name="static")
        
        @app.get("/", response_class=HTMLResponse)
        async def get_html():
            with open("static/index.html") as f:
                return f.read()

if __name__ == "__main__":
    # Check if the static directory exists, create it if not
    if not os.path.exists("static"):
        os.makedirs("static")
    
    # Run the server
    uvicorn.run("web_app:app", host="localhost", port=8000, reload=True)
