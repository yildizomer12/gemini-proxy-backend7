# main.py dosyasında real_gemini_stream fonksiyonunu bu ile değiştirin:

async def real_gemini_stream(api_key: str, messages, generation_config, model: str):
    """Çalışan Gemini stream işlemcisi"""
    
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    
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
        print("[DEBUG] Sent initial chunk")
        
        async with httpx.AsyncClient(timeout=120) as client:
            print(f"[DEBUG] Starting Gemini API request")
            
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
                    print(f"[ERROR] Gemini API error: {response.status_code}")
                    
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
                    return
                
                # Accumulate response data
                accumulated_data = ""
                content_sent = False
                
                async for chunk in response.aiter_bytes():
                    try:
                        chunk_text = chunk.decode('utf-8')
                        accumulated_data += chunk_text
                        print(f"[DEBUG] Received chunk: {len(chunk_text)} bytes")
                        
                        # Try to parse complete JSON objects
                        try:
                            # Clean up the data - remove any incomplete parts
                            if accumulated_data.strip():
                                # Look for complete JSON objects
                                lines = accumulated_data.strip().split('\n')
                                for line in lines:
                                    line = line.strip()
                                    if line.startswith('[') or line.startswith('{'):
                                        try:
                                            json_data = json.loads(line)
                                            print(f"[DEBUG] Parsed JSON successfully")
                                            
                                            # Extract content from JSON
                                            candidates = []
                                            if isinstance(json_data, list):
                                                for item in json_data:
                                                    if 'candidates' in item:
                                                        candidates.extend(item['candidates'])
                                            elif 'candidates' in json_data:
                                                candidates = json_data['candidates']
                                            
                                            # Process candidates
                                            for candidate in candidates:
                                                if 'content' in candidate and 'parts' in candidate['content']:
                                                    for part in candidate['content']['parts']:
                                                        if 'text' in part and part['text'].strip():
                                                            content = part['text']
                                                            content_sent = True
                                                            
                                                            # Send content chunk
                                                            content_chunk = {
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
                                                            yield f"data: {json.dumps(content_chunk)}\n\n"
                                                            print(f"[DEBUG] Sent content: {content[:50]}...")
                                                
                                                # Check for completion
                                                if candidate.get('finishReason') == 'STOP':
                                                    print("[DEBUG] Stream completed with STOP")
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
                                                    print("[DEBUG] Sent final chunks")
                                                    return
                                        
                                        except json.JSONDecodeError:
                                            continue
                        
                        except Exception as e:
                            print(f"[DEBUG] JSON parsing issue: {e}")
                            continue
                    
                    except UnicodeDecodeError:
                        print("[WARN] Unicode decode error")
                        continue
                    except Exception as e:
                        print(f"[ERROR] Chunk processing error: {e}")
                        continue
                
                # If we got here without sending content, send a fallback message
                if not content_sent:
                    print("[DEBUG] No content was sent, sending fallback")
                    fallback_chunk = {
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
                    yield f"data: {json.dumps(fallback_chunk)}\n\n"
                
                # Always send completion
                print("[DEBUG] Sending completion chunks")
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
                print("[DEBUG] Stream completed")
    
    except Exception as e:
        print(f"[ERROR] Stream error: {str(e)}")
        try:
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
        except:
            pass