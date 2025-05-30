import os
import uuid
import time
import json
import asyncio
import base64
from typing import List, Dict, Any, Union, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

# FastAPI app
app = FastAPI(title="Gemini Backend for Vercel - Real Stream")

# API Keys
API_KEYS = [
    "AIzaSyCT1PXjhup0VHx3Fz4AioHbVUHED0fVBP4",
    "AIzaSyArNqpA1EeeXBx-S3EVnP0tzao6r4BQnO0",
    "AIzaSyCXICPfRTnNAFwNQMmtBIb3Pi0pR4SydHg",
    "AIzaSyDiLvp7CU443luErAz3Ck0B8zFdm8UvNRs",
    "AIzaSyBzqJebfbVPcBXQy7r4Y5sVgC499uV85i0",
    "AIzaSyD6AFGKycSp1glkNEuARknMLvo93YbCqH8",
    "AIzaSyBTara5UhTbLR6qnaUI6nyV4wugycoABRM",
    "AIzaSyBI2Jc8mHJgjnXnx2udyibIZyNq8SGlLSY",
    "AIzaSyAcgdqbZsX9UOG4QieFSW7xCcwlHzDSURY",
    "AIzaSyAwOawlX-YI7_xvXY-A-3Ks3k9CxiTQfy4",
    "AIzaSyCJVUeJkqYeLNG6UsF06Gasn4mvMFfPhzw",
    "AIzaSyBFOK0YgaQOg5wilQul0P2LqHk1BgeYErw",
    "AIzaSyBQRsGHOhaiD2cNb5F68hI6BcZR7CXqmwc",
    "AIzaSyCIC16VVTlFGbiQtq7RlstTTqPYizTB7yQ",
    "AIzaSyCIlfHXQ9vannx6G9Pae0rKwWJpdstcZIM",
    "AIzaSyAUIR9gx08SNgeHq8zKAa9wyFtFu00reTM",
    "AIzaSyAST1jah1vAcnLfmofR4DDw0rjYkJXJoWg",
    "AIzaSyAV8OU1_ANXTIvkRooikeNrI1EMR3IbTyQ"
]

# Simple state management
current_key_index = 0
key_usage = {key: 0 for key in API_KEYS}

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
    key_usage[key] = key_usage.get(key, 0) + 1
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

def extract_json_objects(text):
    """Extract JSON objects from text, handling multiple objects and arrays"""
    objects = []
    text = text.strip()
    
    # Handle single JSON object
    if text.startswith('{') and text.endswith('}'):
        try:
            obj = json.loads(text)
            objects.append(obj)
            return objects
        except:
            pass
    
    # Handle JSON array
    if text.startswith('[') and text.endswith(']'):
        try:
            array = json.loads(text)
            if isinstance(array, list):
                objects.extend(array)
            else:
                objects.append(array)
            return objects
        except:
            pass
    
    # Handle multiple JSON objects separated by newlines
    lines = text.split('\n')
    current_obj = ""
    brace_count = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        current_obj += line
        
        # Count braces to detect complete JSON objects
        for char in line:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                
        # When brace count reaches 0, we have a complete object
        if brace_count == 0 and current_obj:
            try:
                obj = json.loads(current_obj)
                objects.append(obj)
                current_obj = ""
            except:
                # If parsing fails, reset and continue
                current_obj = ""
                brace_count = 0
    
    return objects

async def real_gemini_stream(api_key: str, messages, generation_config, model: str):
    """BrokenPipeError'a karşı korumalı Gemini stream işlemcisi"""
    
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    client_disconnected = False
    
    try:
        # Initial chunk with role
        initial_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"
        
        async with httpx.AsyncClient(timeout=120) as client:
            print(f"[DEBUG] Sending request to Gemini API with model: {model}")
            
            async with client.stream(
                'POST',
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent",
                json={
                    "contents": messages,
                    "generationConfig": generation_config
                },
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                }
            ) as response:
                
                if response.status_code != 200:
                    error_text = await response.aread()
                    print(f"[ERROR] Gemini API error - Status: {response.status_code}, Response: {error_text.decode()}")
                    
                    if not client_disconnected:
                        try:
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
                            yield "data: [DONE]\n\n"
                        except GeneratorExit:
                            client_disconnected = True
                            return
                    return
                
                # Buffer for accumulating response data
                response_buffer = ""
                last_sent_time = time.time()
                
                async for chunk in response.aiter_bytes():
                    # Early exit if client disconnected
                    if client_disconnected:
                        print("[DEBUG] Client disconnected, stopping stream")
                        return
                        
                    try:
                        # Decode chunk and add to buffer
                        chunk_text = chunk.decode('utf-8')
                        response_buffer += chunk_text
                        
                        print(f"[DEBUG] Buffer content: {response_buffer[:200]}...")  # First 200 chars
                        
                        # Try to extract complete JSON objects from buffer
                        json_objects = extract_json_objects(response_buffer)
                        
                        if json_objects:
                            # Process each JSON object
                            for json_obj in json_objects:
                                if client_disconnected:
                                    return
                                    
                                if 'candidates' in json_obj:
                                    candidate = json_obj['candidates'][0]
                                    
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part and part['text'].strip():
                                                content = part['text']
                                                
                                                try:
                                                    # Create OpenAI format chunk
                                                    openai_chunk = {
                                                        'id': chunk_id,
                                                        'object': 'chat.completion.chunk',
                                                        'created': created_time,
                                                        'model': model,
                                                        'choices': [{
                                                            'index': 0,
                                                            'delta': {'content': content},
                                                            'finish_reason': None
                                                        }]
                                                    }
                                                    
                                                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                                                    last_sent_time = time.time()
                                                    print(f"[DEBUG] Sent content chunk: {content[:50]}...")
                                                except GeneratorExit:
                                                    client_disconnected = True
                                                    print("[DEBUG] Client disconnected during content sending")
                                                    return
                                    
                                    # Check for finish reason
                                    if candidate.get('finishReason') == 'STOP':
                                        if not client_disconnected:
                                            try:
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
                                                print("[DEBUG] Stream completed normally")
                                            except GeneratorExit:
                                                client_disconnected = True
                                                print("[DEBUG] Client disconnected during completion")
                                        return
                            
                            # Clear buffer after successful processing
                            response_buffer = ""
                        
                        # Send keep-alive if no data sent for 10 seconds
                        current_time = time.time()
                        if current_time - last_sent_time >= 10 and not client_disconnected:
                            try:
                                keep_alive_chunk = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": ""},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(keep_alive_chunk)}\n\n"
                                last_sent_time = current_time
                                print("[DEBUG] Sent keep-alive chunk")
                            except GeneratorExit:
                                client_disconnected = True
                                print("[DEBUG] Client disconnected during keep-alive")
                                return
                    
                    except UnicodeDecodeError:
                        print("[WARN] Unicode decode error, skipping chunk")
                        continue
                    except GeneratorExit:
                        client_disconnected = True
                        print("[DEBUG] Client disconnected during processing")
                        return
                    except Exception as e:
                        print(f"[ERROR] Chunk processing error: {e}")
                        continue
                
                # Handle any remaining buffer content
                if response_buffer.strip() and not client_disconnected:
                    print(f"[DEBUG] Processing remaining buffer: {response_buffer}")
                    try:
                        json_objects = extract_json_objects(response_buffer)
                        for json_obj in json_objects:
                            if client_disconnected:
                                return
                                
                            if 'candidates' in json_obj:
                                candidate = json_obj['candidates'][0]
                                if 'content' in candidate and 'parts' in candidate['content']:
                                    for part in candidate['content']['parts']:
                                        if 'text' in part and part['text'].strip():
                                            content = part['text']
                                            try:
                                                openai_chunk = {
                                                    'id': chunk_id,
                                                    'object': 'chat.completion.chunk',
                                                    'created': created_time,
                                                    'model': model,
                                                    'choices': [{
                                                        'index': 0,
                                                        'delta': {'content': content},
                                                        'finish_reason': None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                            except GeneratorExit:
                                                client_disconnected = True
                                                return
                    except Exception as e:
                        print(f"[ERROR] Final buffer processing error: {e}")
    
    except asyncio.CancelledError:
        print("[INFO] Stream cancelled by client")
        client_disconnected = True
        return
    except GeneratorExit:
        print("[INFO] Client disconnected (GeneratorExit)")
        client_disconnected = True
        return
    except Exception as e:
        print(f"[ERROR] Stream error: {str(e)}")
        if not client_disconnected:
            try:
                error_chunk = {
                    'id': chunk_id,
                    'object': 'chat.completion.chunk',
                    'created': created_time,
                    'model': model,
                    'choices': [{
                        'index': 0,
                        'delta': {'content': f"Stream error: {str(e)}"},
                        'finish_reason': 'error'
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            except GeneratorExit:
                client_disconnected = True
                return
    
    finally:
        # Ensure proper termination only if client is still connected
        if not client_disconnected:
            try:
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
                print("[DEBUG] Stream terminated normally")
            except GeneratorExit:
                print("[DEBUG] Client disconnected during final cleanup")
                pass

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        print(f"[DEBUG] Received request: model={request.model}, stream={request.stream}, messages={len(request.messages)}")
        
        gemini_messages = convert_messages(request.messages)
        generation_config = {"temperature": request.temperature}
        if request.max_tokens:
            generation_config["max_output_tokens"] = request.max_tokens
        
        api_key = get_next_key()
        print(f"[DEBUG] Using API key: {api_key[:10]}...")
        
        if request.stream:
            async def safe_stream_wrapper():
                """Stream wrapper with proper error handling"""
                try:
                    async for chunk in real_gemini_stream(api_key, gemini_messages, generation_config, request.model):
                        yield chunk
                except asyncio.CancelledError:
                    print("[INFO] Stream cancelled in wrapper")
                    return
                except GeneratorExit:
                    print("[INFO] Generator exit in wrapper")
                    return
                except Exception as e:
                    print(f"[ERROR] Stream wrapper error: {str(e)}")
                    # Send error chunk if possible
                    try:
                        error_chunk = {
                            'id': f"chatcmpl-{uuid.uuid4().hex}",
                            'object': 'chat.completion.chunk',
                            'created': int(time.time()),
                            'model': request.model,
                            'choices': [{
                                'index': 0,
                                'delta': {'content': f"Error: {str(e)}"},
                                'finish_reason': 'error'
                            }]
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                    except:
                        pass
            
            return StreamingResponse(
                safe_stream_wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Transfer-Encoding": "chunked",
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
                
                if not response.is_success:
                    print(f"[ERROR] Non-stream API error: {response.status_code} - {response.text}")
                    raise HTTPException(status_code=response.status_code, detail="Gemini API error")
                
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
                
    except Exception as e:
        print(f"[ERROR] Request processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "key_usage": sum(key_usage.values()),
        "total_keys": len(API_KEYS),
        "current_key_index": current_key_index,
        "stream_version": "v2_improved"
    }

@app.head("/health")
async def head_health():
    return {}

@app.get("/debug")
async def debug_info():
    return {
        "available_endpoints": [
            {"path": "/", "methods": ["GET", "HEAD"]},
            {"path": "/health", "methods": ["GET", "HEAD"]},
            {"path": "/v1/chat/completions", "methods": ["POST", "OPTIONS"]},
            {"path": "/debug", "methods": ["GET"]}
        ],
        "server_info": {
            "python_version": "3.12",
            "fastapi_running": True,
            "vercel_deployment": True
        }
    }

@app.api_route("/test", methods=["GET", "POST", "OPTIONS"])
async def test_endpoint(request: Request):
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": dict(request.headers),
        "message": f"Test endpoint received {request.method} request"
    }

@app.get("/")
async def root():
    return {
        "message": "Gemini Proxy Server",
        "version": "2.0",
        "endpoints": ["/v1/chat/completions", "/health"],
        "status": "running"
    }

@app.head("/")
async def head_root():
    return {}

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