from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import os
from uuid import uuid4
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# init env keys
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="LangChain LLM API")


host_origin = os.getenv('ORIGIN')

origins = [
    "https://tangpt.tanflix.me/",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
]

if host_origin :
    origins.append(host_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Model for the chat request
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    query: str

# Model for the chat response
class ChatResponse(BaseModel):
    conversation_id: str
    response: str

class GraphResponse(BaseModel):
    graph : str = Field( description="graph code for the given user query or description")
    direct_response : str = Field( description="direct response only when user query cannot be converted into graph")

# Store conversation histories
conversation_histories: Dict[str, ChatMessageHistory] = {}




# System prompts
SYSTEM_PROMPT = """
You are a specialized AI assistant that converts text descriptions into diagram code. 
Your primary function is to interpret user queries and generate appropriate diagram code (Mermaid, PlantUML, etc.) based on their descriptions.

## Response Format
Your responses should be structured as follows:
- always try to Provide response in diagram code in the "graph" field and leave "direct_response" empty.
- If the user query cannot reasonably be converted to a diagram: Provide a helpful explanation in the "direct_response" field and leave "graph" empty.


## Example Diagram Syntaxes
- For flowcharts, use Mermaid syntax: `graph TD`

Always strive to produce the most accurate and useful diagram representation of the user's description.
"""

# Default prompt style
DEFAULT_PROMPT_STYLE = "default"



# Initialize the LLM
def llm_chain():
    
    model =  ChatOpenAI(
        model="gpt-4o-mini" 
    )
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "query: {query}"),
            ]
        )
    return prompt | model.with_structured_output(GraphResponse)

@app.post("/chat")
async def chat(request: ChatRequest):
    # Get or create conversation history
    conversation_id = request.conversation_id or str(uuid4())
    
    if conversation_id not in conversation_histories:
        # Create new conversation history
        conversation_histories[conversation_id] = ChatMessageHistory()
        # Add system message to new conversations
        
    
    # Add user query to history
    conversation_histories[conversation_id]
    
    # Get message history
    messages = conversation_histories[conversation_id].messages
    
    # Add system message for all conversations
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))
    
    try:
        # Call LLM
        chain = llm_chain()
        ai_response = chain.invoke(
            {
                "messages": messages,
                "query": request.query,
            }
        )
        
        # Add AI response to history
        conversation_histories[conversation_id].add_ai_message(ai_response.json())
        return ai_response
        
        #return ChatResponse(
        #    conversation_id=conversation_id,
        #    response=ai_response.json()
        #)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")


@app.get("/conversations")
async def list_conversations():
    return {
        "conversation_ids": list(conversation_histories.keys())
    }

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    if conversation_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Format messages for readability
    formatted_messages = []
    for msg in conversation_histories[conversation_id].messages:
        if isinstance(msg, SystemMessage):
            formatted_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            formatted_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_messages.append({"role": "assistant", "content": msg.content})
    
    return {
        "conversation_id": conversation_id,
        "messages": formatted_messages
    }

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if conversation_id not in conversation_histories:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    del conversation_histories[conversation_id]
    return {"status": "Conversation deleted"}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)