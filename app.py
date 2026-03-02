import asyncio
import os
import logging

from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

from utils.utility import _http
from config.logging_config import setup_logging

# 1. Load environment variables from .env file
load_dotenv()

# 2. Initialize the global config
# setup_logging()

OLLAMA_API_URL          = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
MAX_CONTEXT_MESSAGES    = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))

# Model generation options
DEFAULT_LLM_MODEL       = os.getenv("DEFAULT_LLM_MODEL", "qwen3-vl:2b")
LLM_TEMPERATURE         = float(os.getenv("LLM_TEMPERATURE", "0.7"))            # Controls randomness in generation (higher = more random)
LLM_NUM_CTX             = int(os.getenv("LLM_NUM_CTX", "2048"))                 # Maximum context length (number of tokens) the model can consider
LLM_NUM_PREDICT         = int(os.getenv("LLM_NUM_PREDICT", "256"))              # Maximum number of tokens to generate in the response (None means no limit)
LLM_NUM_THREAD          = int(os.getenv("LLM_NUM_THREAD", str(os.cpu_count() or 4)))
LLM_STOP                = os.getenv("LLM_STOP", "None")                         # Stop sequences that will halt generation
LLM_STREAM              = os.getenv("LLM_STREAM", "False").lower() == "true"    # Enable/disable streaming responses
LLM_THINK               = os.getenv("LLM_THINK", "False").lower() == "true"     # Enable/disable thinking mode


class ChatMessage(BaseModel):
    """Represents a single message in a chat conversation.
    
    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
    """
    role: Literal["system", "user", "assistant"] = Field(
        description="The role of the message sender."
    )
    content: str = Field(
        min_length=1,
        max_length=10000,
        description="The content of the message. Must be between 1 and 10000 characters."
    )


class ChatRequest(BaseModel):
    """Represents a request for chat completion.
    
    Attributes:
        model: The name of the model to use for generating the chat completion.
        messages: A list of messages in the conversation.
    """
    model: str = Field(
        default=DEFAULT_LLM_MODEL,
        min_length=1,
        max_length=200,
        description="The name of the model to use for generating the chat completion."
    )
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="A list of messages in the conversation."
    )


app = FastAPI(
    title="LocalGPT Backend",
    description="A local implementation for the using local LLM models.",
    version="1.0.0"
)

logger = logging.getLogger(__name__)

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):

    # Sanitize the messages
    sanitized_messages = [
        ChatMessage(role=message.role, content=message.content.strip())
        for message in request.messages if message.content.strip()
    ]

    logger.info(
        "Started processing chat request. messages_received=%d sanitized_messages=%d model=%s",
        len(request.messages),
        len(sanitized_messages),
        request.model
    )

    # If there are no valid messages after sanitization, return an error response
    if not sanitized_messages:
        logger.warning("No valid messages found in the request after sanitization.")
        return {
            "error": {
                "message": "No valid messages provided. Please ensure that your messages contain non-empty content.",
                "type": "invalid_request_error",
                "param": None,
                "code": 400
            }
        }

    # Only provide the last N messages to the model to avoid overwhelming it with too much context
    trimmed_messages = sanitized_messages[-MAX_CONTEXT_MESSAGES:]

    # Create the payload for the local LLM model. for ollama
    payload = {
        "model": request.model,
        "stream": LLM_STREAM,
        "think": LLM_THINK,
        "messages": [ message.model_dump() for message in trimmed_messages],
        "options": {
            "temperature": LLM_TEMPERATURE,
            "num_ctx": LLM_NUM_CTX,
            "num_predict": LLM_NUM_PREDICT,
            "num_thread": LLM_NUM_THREAD,
            "stop": LLM_STOP if LLM_STOP != "None" else None
        }
    }

    ollama_chat_url = f"{OLLAMA_API_URL}/api/chat"
    logger.debug("Sending request to local LLM model at %s with payload: %s", ollama_chat_url, payload)

    status, response, resp_headers = _http(url=ollama_chat_url, method="POST", body=payload)

    if status != 200:
        logger.error("Failed to get response from local LLM model. Status: %d, Response: %s", status, response)
        return {
            "error": {
                "message": f"Failed to get response from local LLM model. Status: {status}",
                "type": "model_error",
                "param": None,
                "code": status
            }
        }
    
    return response


@app.get("/stream")
async def stream() -> StreamingResponse:

    async def generator(r: int):
        for i in range(r):
            yield f"data: {i}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(generator(20), media_type="text/event-stream")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}