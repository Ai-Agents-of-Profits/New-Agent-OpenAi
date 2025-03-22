# Conversation State Implementation

## Overview
This document explains how conversation state is implemented in the crypto trading assistant using a manual conversation history approach.

## Implementation Details

The conversation state is maintained using a `ConversationHistory` class that stores the sequence of messages between the user and agent. This allows us to create threaded conversations with full context awareness.

### Key Components:

1. **ConversationHistory Class**:
   - Stores an array of message objects with roles (user/assistant) and content
   - Provides methods to add new messages and retrieve the full history
   - Maintains the context across multiple interactions

2. **Main CLI Interface**:
   - Initializes and maintains a ConversationHistory instance
   - Passes the full message history for follow-up questions
   - Provides a "new topic" or "reset" command to clear conversation history

3. **API Interface**:
   - The `handle_user_message` function now accepts and returns a ConversationHistory object
   - This allows API consumers to maintain conversation threads appropriately

### Benefits

- **Complete Context Management**: Full conversation history is available for complex follow-up questions
- **More Control Over Context**: Ability to selectively manage which parts of conversation are included
- **Simple Implementation**: Straightforward approach using standard data structures
- **Improved User Experience**: The agent remembers previous interactions naturally

### Example Conversation Flow

1. User sends first query: "What's the current price of BTC/USDT?"
   - Agent processes the request directly
   - Adds both the request and response to ConversationHistory

2. User follows up: "What about ETH?"
   - System passes the full message history
   - Agent understands the context (that "ETH" refers to the price of ETH/USDT)
   - Adds this interaction to the conversation history

3. User asks: "How has it changed in the past week?"
   - System passes the updated message history
   - Agent understands "it" refers to ETH price from previous interaction
   - Provides appropriate technical analysis with context

### Conversation Reset Conditions

The conversation context is reset (conversation history cleared) when:
- A guardrail is triggered
- An error occurs during processing
- The user explicitly enters "new topic" or "reset"

### Manual Reset

Users can type "new topic" or "reset" at any time to clear the conversation history and start a fresh context. This is useful when switching to a completely different topic or when the conversation has accumulated too much context.
