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

from enhanced_main import setup_crypto_agents, handle_user_message
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

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

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

# WebSocket endpoint for chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            user_message = data_json.get("message", "")
            if not user_message:
                continue
                
            # Send acknowledgment
            await manager.send_personal_message(
                json.dumps({"type": "processing", "data": "Processing your request..."}),
                websocket
            )
            
            # Process the message with the agent
            try:
                logger.info(f"Processing request: {user_message}")
                
                # Run the agent
                result = await Runner.run(app.state.agent, user_message)
                
                # Get the agent's response
                response = result.final_output
                
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
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "data": f"Error processing your request: {str(e)}"
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
        user_id = data.get("user_id", "default_user")
        
        # Process the message using our enhanced_main module
        response = await handle_user_message(user_message, user_id)
        
        return {"response": response}
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
