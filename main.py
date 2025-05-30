import os
import uuid
import time
import json
import asyncio
import base64
from typing import List, Dict, Any, Union, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import httpx

# FastAPI app
app = FastAPI()

# API Keys
API_KEYS = [
    "AIzaSyCT1PXjhup0VHx3Fz4AioHbVUHED0fVBP4",
    "AIzaSyArNqpA1EeeXBx-S3EVnP0tzao6r4BQnO0",
    "AIzaSyCXICPfRTnNAFwNQMmtBIb3Pi0pR4SydHg",
    "AIzaSyDiLvp7CU443luErAz3Ck0B8zFdm8UvNRs",
    "AIzaSyBzqJebfbVPcBXQy7r4Y5sVgC499uV85i0"
]

# Simple state management
current_key_index = 0

# Pydantic Models
class ImageUrl(BaseModel):
    url: str

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False

def get_next_key():
    """Simple round-robin key selection"""
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return key

def process_content(content):
    """Convert OpenAI format to Gemini format"""
    if isinstance(content, str):
        return [{"text": content}]
    
    parts = []
    for item in content:
        if item.type == "text" and item.text:
            parts.append({"text": item.text})
        elif item.type == "image_url" and item.image_url:
            try:
                if item.image_url.url.startswith("data:"):
                    header, base64_data = item.image_url.url.split(",", 1)
                    mime_type = header.split(";")[0].split(":")[1]
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_data
                        }
                    })
            except:
                parts.append({"text": "[Image processing error]"})
    
    return parts or [{"text": ""}]

def convert_messages(messages):
    """Convert OpenAI messages to Gemini format"""
    return [
        {
            "role": "user" if msg.role == "user" else "model",
            "parts": process_content(msg.content)
        }
        for msg in messages
    ]

async def simple_gemini_stream(api_key: str, messages, generation_config, model: str):
    """Basit stream generator"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    
    try:
        # Initial chunk
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created_time, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
        
        # Make non-streaming request to Gemini and simulate streaming
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                json={
                    "contents": messages,
                    "generationConfig": generation_config
                },
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                text = ""
                if result.get("candidates"):
                    candidate = result["candidates"][0]
                    if candidate.get("content", {}).get("parts"):
                        text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
                
                if text:
                    # Split text into chunks and stream
                    words = text.split()
                    current_text = ""
                    
                    for i, word in enumerate(words):
                        current_text += word + " "
                        
                        # Send chunk every few words
                        if i % 3 == 0 or i == len(words) - 1:
                            chunk = {
                                'id': chunk_id,
                                'object': 'chat.completion.chunk',
                                'created': created_time,
                                'model': model,
                                'choices': [{
                                    'index': 0,
                                    'delta': {'content': current_text},
                                    'finish_reason': None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            current_text = ""
                            await asyncio.sleep(0.1)  # Small delay for streaming effect
                else:
                    # Fallback message
                    chunk = {
                        'id': chunk_id,
                        'object': 'chat.completion.chunk',
                        'created': created_time,
                        'model': model,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': "I received your message. How can I help you?"},
                            'finish_reason': None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Error handling
                error_chunk = {
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': model,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': f"API Error: {response.status_code}"},
                        'finish_reason': 'error'
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
        
        # Final chunk
        final_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': created_time,
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {},
                'finish_reason': 'stop'
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        print(f"Stream error: {e}")
        error_chunk = {
            'id': chunk_id,
            'object': 'chat.completion.chunk',
            'created': created_time,
            'model': model,
            'choices': [{
                'index': 0,
                'delta': {'content': f"Error: {str(e)}"},
                'finish_reason': 'error'
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini Proxy is running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": "simple"
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        gemini_messages = convert_messages(request.messages)
        generation_config = {"temperature": request.temperature}
        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens
        
        api_key = get_next_key()
        
        if request.stream:
            return StreamingResponse(
                simple_gemini_stream(api_key, gemini_messages, generation_config, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Access-Control-Allow-Origin": "*",
                    "Content-Type": "text/event-stream"
                }
            )
        else:
            # Non-streaming response
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{request.model}:generateContent",
                    json={
                        "contents": gemini_messages,
                        "generationConfig": generation_config
                    },
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = ""
                    if result.get("candidates"):
                        candidate = result["candidates"][0]
                        if candidate.get("content", {}).get("parts"):
                            text = "".join(part.get("text", "") for part in candidate["content"]["parts"])
                    
                    if not text:
                        text = "No response generated"
                    
                    return {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    }
                else:
                    raise HTTPException(status_code=response.status_code, detail="Gemini API error")
                
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def cors_handler(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

@app.options("/{path:path}")
async def options_handler():
    return {"message": "OK"}