# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import readline
import sys
import uuid
from datetime import datetime

import httpx
from httpx import Client
from httpx_curl_cffi import CurlTransport


def format_as_markdown(text):
    """Format text as markdown for better readability"""
    # Handle multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Add proper spacing for headings
    text = re.sub(r'(^|\n)(#{1,6}\s)', r'\1\n\2', text)

    # Add proper spacing for list items
    text = re.sub(r'(^|\n)([*+-]\s)', r'\1\n\2', text)
    text = re.sub(r'(^|\n)(\d+\.\s)', r'\1\n\2', text)

    # Clean up extra newlines at the beginning
    text = text.lstrip('\n')

    return text


def format_chunk_realtime(chunk, accumulated_text):
    """Format text chunk in real-time for better display"""
    formatted_chunk = chunk

    # Handle newlines - convert double newlines to proper spacing
    if '\n\n' in chunk:
        formatted_chunk = re.sub(r'\n\n+', '\n\n', chunk)

    # Check for markdown elements and add proper spacing
    lines = formatted_chunk.split('\n')
    result_lines = []

    for i, line in enumerate(lines):
        # Handle headings
        if line.strip().startswith('#'):
            # Add newline before heading if not at start and previous line isn't empty
            if result_lines and result_lines[-1].strip() != '':
                result_lines.append('')
            result_lines.append(line)
            # Add newline after heading
            if i < len(lines) - 1 and lines[i + 1].strip() != '':
                result_lines.append('')
        # Handle list items
        elif re.match(r'^\s*[*+-]\s', line) or re.match(r'^\s*\d+\.\s', line):
            # Add newline before list if not at start and previous line isn't empty
            if (result_lines and
                    result_lines[-1].strip() != '' and not
                    re.match(r'^\s*[*+-]\s', result_lines[-1]) and not
                    re.match(r'^\s*\d+\.\s', result_lines[-1])):
                result_lines.append('')
            result_lines.append(line)
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def check_cookie_expiry(cookie):
    """Check if the cookie contains required authentication tokens"""
    try:
        # Check for required authentication tokens
        required_tokens = ['arena-auth-prod-v1.0', '__cf_bm', 'cf_clearance']
        missing_tokens = []
        
        for token in required_tokens:
            if token not in cookie:
                missing_tokens.append(token)
        
        if missing_tokens:
            return False, f"Cookie缺少必要的认证token: {', '.join(missing_tokens)}"
        
        # Check if cookie looks fresh (contains recent cf_clearance)
        if '__cf_bm=' in cookie:
            # Extract timestamp from __cf_bm token (rough estimation)
            cf_bm_start = cookie.find('__cf_bm=') + len('__cf_bm=')
            cf_bm_end = cookie.find('-', cf_bm_start)
            if cf_bm_end > cf_bm_start:
                try:
                    # This is a rough check, __cf_bm format may vary
                    return True, "Cookie包含必要的认证信息"
                except:
                    pass
        
        return True, "Cookie格式正确，包含基本认证信息"
        
    except Exception as e:
        return True, f"Cookie验证时出错: {e}"


def get_cookie():
    """Get cookie from command line argument or file"""
    parser = argparse.ArgumentParser(description='HTTP request with cookie')
    parser.add_argument('--cookie', type=str, help='Cookie string')
    args, _ = parser.parse_known_args()

    # If cookie provided via command line, use it
    if args.cookie:
        return args.cookie

    # Otherwise, try to read from cookie.txt file
    try:
        if os.path.exists('cookie.txt'):
            with open('cookie.txt', 'r', encoding='utf-8') as f:
                cookie = f.read().strip()
                if cookie:
                    # Basic cookie validation
                    if 'arena-auth-prod' in cookie:
                        # Check cookie expiry
                        is_valid, message = check_cookie_expiry(cookie)
                        print(f"Cookie状态: {message}")
                        if not is_valid:
                            print("建议: 请更新cookie.txt文件中的cookie")
                        return cookie
                    else:
                        print("警告: cookie.txt中的cookie可能无效，缺少必要的认证信息")
                        return cookie
                else:
                    print("错误: cookie.txt文件为空")
        else:
            print("错误: 未找到cookie.txt文件，请创建该文件并添加有效的cookie")
    except Exception as e:
        print(f"读取cookie.txt文件时出错: {e}")

    # If no cookie found, return None
    return None


def transport_factory(proxy: httpx.Proxy | None = None) -> httpx.BaseTransport:
    return CurlTransport(  # or AsyncCurlTransport
        proxy=proxy,
        verify=False,  # and other custom options
    )


def make_request(content, session_id=None, conversation_history=None):
    # Use provided content for the request
    if conversation_history is None:
        conversation_history = []

    # Get cookies from command line or file
    cookies = get_cookie()
    if not cookies:
        raise ValueError(
            "Cookie is required but not provided. Please provide cookie via --cookie argument or in cookie.txt file.")

    # Headers
    headers = {
        'Host': 'lmarena.ai',
        'Connection': 'keep-alive',
        'sec-ch-ua-full-version-list': '"Chromium";v="140.0.7339.80", "Not=A?Brand";v="24.0.0.0", "Google Chrome";v="140.0.7339.80"',
        'sec-ch-ua-platform': '"macOS"',
        'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
        'sec-ch-ua-bitness': '"64"',
        'sec-ch-ua-model': '""',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-arch': '"arm"',
        'sec-ch-ua-full-version': '"140.0.7339.80"',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
        'Content-Type': 'text/plain;charset=UTF-8',
        'sec-ch-ua-platform-version': '"26.0.0"',
        'Accept': '*/*',
        'Origin': 'https://lmarena.ai',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Referer': 'https://lmarena.ai/?mode=direct',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Cookie': cookies,
    }

    # JSON data
    # Generate new UUIDs for each request
    if session_id is None:
        session_id = str(uuid.uuid4())
        url = "https://lmarena.ai/nextjs-api/stream/create-evaluation"
    else:
        url = "https://lmarena.ai/nextjs-api/stream/post-to-evaluation/" + session_id

    model_a_message_id = str(uuid.uuid4())
    user_message_id = str(uuid.uuid4())
    messages = []
    for msg in conversation_history:
        messages.append(msg)

    # Add current user message
    user_msg = {
        "id": user_message_id,
        "role": "user",
        "content": content,
        "experimental_attachments": [],
        "parentMessageIds": [conversation_history[-1]["id"]] if conversation_history else [],
        "participantPosition": "a",
        "evaluationSessionId": session_id,
        "modelId": None,
        "status": "pending",
        "failureReason": None,
    }
    messages.append(user_msg)
    messages.append({
        "id": model_a_message_id,
        "role": "assistant",
        "content": "",
        "experimental_attachments": [],
        "parentMessageIds": [
            user_message_id
        ],
        "participantPosition": "a",
        "modelId": "983bc566-b783-4d28-b24c-3c8b08eb1086",
        "evaluationSessionId": session_id,
        "status": "pending",
        "failureReason": None
    })
    data = {
        "id": session_id,
        "mode": "direct",
        # gpt-5-high
        # "modelAId": "983bc566-b783-4d28-b24c-3c8b08eb1086",
        # claude-4-1-opus
        "modelAId": "96ae95fd-b70d-49c3-91cc-b58c7da1090b",
        "userMessageId": user_message_id,
        "modelAMessageId": model_a_message_id,
        "messages": messages,
        "modality": "chat"
    }

    proxy_url = "http://127.0.0.1:7890"
    print("\nthinking...")
    try:
        with Client(transport=CurlTransport(
                proxy=proxy_url, impersonate="chrome", default_headers=True,
                verify=False)) as client:
            with client.stream(
                    method="POST",
                    url=url,
                    headers=headers,
                    json=data,
                    timeout=300.0
            ) as response:
                # Check if response has an error status code
                if response.status_code >= 400:
                    error_msg = f"Error: Received HTTP {response.status_code} response"
                    if response.status_code == 403:
                        error_msg += "\n可能的原因：\n1. Cookie已过期，请更新cookie.txt文件\n2. 请求被服务器拒绝，请检查请求头信息\n3. IP被限制访问"
                        error_msg += "\n\n获取新cookie的步骤："
                        error_msg += "\n1. 打开浏览器访问 https://lmarena.ai"
                        error_msg += "\n2. 登录你的账户"
                        error_msg += "\n3. 按F12打开开发者工具"
                        error_msg += "\n4. 在Network标签页中刷新页面"
                        error_msg += "\n5. 找到任意请求，复制Cookie请求头的值"
                        error_msg += "\n6. 将完整的Cookie值粘贴到cookie.txt文件中"
                    elif response.status_code == 401:
                        error_msg += "\n认证失败，请检查cookie是否正确"
                    elif response.status_code == 429:
                        error_msg += "\n请求过于频繁，请稍后再试"
                    
                    try:
                        response_text = response.text
                        if response_text:
                            error_msg += f"\n服务器响应: {response_text[:200]}..."
                    except:
                        pass
                    
                    print(error_msg)
                    return None

                # Process SSE stream with typewriter effect and markdown formatting
                accumulated_text = ""
                assistant_response = ""

                # Create assistant message that will be populated during streaming
                assistant_msg = {
                    "id": str(uuid.uuid4()),
                    "evaluationSessionId": session_id,
                    "parentMessageIds": [user_message_id],
                    "content": "",
                    "modelId": "983bc566-b783-4d28-b24c-3c8b08eb1086",
                    "status": "success",
                    "failureReason": None,
                    "metadata": None,
                    "createdAt": datetime.now().isoformat() + "+00:00",
                    "updatedAt": datetime.now().isoformat() + "+00:00",
                    "role": "assistant",
                    "experimental_attachments": [],
                    "participantPosition": "a"
                }

                for line in response.iter_lines():
                    if line.strip():
                        # Parse SSE data with a0: prefix for text chunks
                        if line.startswith("a0:"):
                            # Extract text content after a0: prefix
                            text_chunk = line[3:]  # Remove "a0:" prefix
                            # Remove quotes if present
                            if text_chunk.startswith('"') and text_chunk.endswith('"'):
                                text_chunk = text_chunk[1:-1]

                            # Convert literal \n to actual newlines
                            text_chunk = text_chunk.replace('\\n', '\n')

                            # Format chunk in real-time for better markdown display
                            assistant_response += text_chunk
                            formatted_chunk = format_chunk_realtime(text_chunk, accumulated_text)
                            accumulated_text += text_chunk

                            # Print the formatted chunk for typewriter effect
                            print(formatted_chunk, end='', flush=True)

                        # Parse completion data with ad: prefix
                        elif line.startswith("ad:"):
                            completion_data = line[3:]  # Remove "ad:" prefix
                            try:
                                json_data = json.loads(completion_data)
                                if json_data.get("finishReason") == "stop":
                                    assistant_msg["content"] = assistant_response
                                    assistant_msg["status"] = "success"
                                    assistant_msg["updatedAt"] = datetime.now().isoformat() + "+00:00"
                                    break
                            except json.JSONDecodeError:
                                pass

                        # Handle other SSE formats
                        elif line.startswith("data: "):
                            data_content = line[6:]  # Remove "data: " prefix
                            if data_content.strip() == "[DONE]":
                                break

                user_msg["createdAt"] = datetime.now().isoformat() + "+00:00"
                user_msg["updatedAt"] = datetime.now().isoformat() + "+00:00"
                user_msg["metadata"] = None
                # Return session info and updated conversation history
                conversation_history.extend([user_msg, assistant_msg])
                return {
                    "session_id": session_id,
                    "conversation_history": conversation_history,
                    "assistant_response": assistant_response,
                    "response": response
                }

    except httpx.RequestError as e:
        print(f"Request error: {e}")
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def save_conversation_history(conv_history, sess_id):
    """Save conversation history to chat directory"""
    if not conv_history:
        return

    # Create chat directory if it doesn't exist
    chat_dir = "chat"
    os.makedirs(chat_dir, exist_ok=True)

    # Generate filename with timestamp
    filename = f"{sess_id}.json"
    filepath = os.path.join(chat_dir, filename)

    # Prepare conversation data
    chat_data = {
        "session_id": sess_id,
        "timestamp": datetime.now().isoformat(),
        "conversation_history": conv_history,
        "message_count": len(conv_history)
    }

    # Save to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        print(f"对话已保存到: {filename}")
    except Exception as e:
        print(f"保存对话失败: {e}")


def handle_special_commands(user_input, conv_history, sess_id):
    """Handle special commands like quit, exit, clear"""
    if user_input.lower() in ['quit', 'exit', '退出']:
        # Save conversation before exit
        if conv_history and sess_id:
            save_conversation_history(conv_history, sess_id)
        print("再见！")
        return "exit"
    elif user_input.lower() in ['clear', '清空']:
        # Save conversation before clearing
        if conv_history and sess_id:
            save_conversation_history(conv_history, sess_id)
        print("对话历史已清空")
        return "clear"
    return None


def setup_readline():
    """Setup readline for command history and editing"""
    # Enable history file in current directory
    history_file = ".chatbot_history"
    try:
        readline.read_history_file(history_file)
    except (FileNotFoundError, PermissionError, OSError) as e:
        # 如果文件不存在、没有权限或格式错误，忽略错误
        if isinstance(e, OSError) and e.errno == 22:  # Invalid argument
            # 历史文件可能损坏，删除并重新创建
            try:
                os.remove(history_file)
                print(f"历史文件已损坏，已重新创建: {history_file}")
            except:
                pass
        pass  # History file doesn't exist yet or permission denied
    
    # Set history length
    readline.set_history_length(1000)
    
    # Save history on exit
    import atexit
    def save_history():
        try:
            readline.write_history_file(history_file)
        except PermissionError:
            pass  # Ignore permission errors when saving
    atexit.register(save_history)


def conversation_loop(current_session_id=None, current_conversation_history=None):
    """Main conversation loop"""
    if current_session_id is None:
        current_session_id = None
    if current_conversation_history is None:
        current_conversation_history = []
    
    # Setup readline for history support
    setup_readline()

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue
            command_result = handle_special_commands(user_input, current_conversation_history, current_session_id)
            if command_result == "exit":
                break
            elif command_result == "clear":
                current_session_id = None
                current_conversation_history = []
                continue
            print("\n助手: ", end='', flush=True)
            result = make_request(user_input, current_session_id, current_conversation_history)

            if result:
                current_session_id = result["session_id"]
                current_conversation_history = result["conversation_history"]
            else:
                print("请求失败，请重试")

        except KeyboardInterrupt:
            print("\n\n对话已中断")
            # Save conversation before exit
            if current_conversation_history and current_session_id:
                save_conversation_history(current_conversation_history, current_session_id)
            break
        except Exception as e:
            print(f"\n发生错误: {e}")


def start_conversation():
    """Start a multi-turn conversation session"""
    print("输入 'clear' 清空对话历史")
    print("提示: 使用上下键可以浏览历史输入")
    print("-" * 50)

    conversation_loop()


if __name__ == "__main__":
    # Check if there's a command line argument for starting conversation with initial message
    if len(sys.argv) > 1:
        initial_content = sys.argv[1]
        print("输入 'clear' 清空对话历史")
        print("-" * 50)

        main_session_id = None
        main_conversation_history = []

        # Send initial message
        print("\n你: " + initial_content)
        print("\n助手: ", end='', flush=True)
        result = make_request(initial_content, main_session_id, main_conversation_history)

        if result:
            main_session_id = result["session_id"]
            main_conversation_history = result["conversation_history"]

        # Continue with interactive conversation
        conversation_loop(main_session_id, main_conversation_history)
    else:
        # Start multi-turn conversation
        start_conversation()
