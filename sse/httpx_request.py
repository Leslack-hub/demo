import copy
import sys
import uuid
import re
import json
import os
from datetime import datetime

import httpx

from httpx import Client, AsyncClient
from httpx_curl_cffi import CurlTransport, AsyncCurlTransport, CurlOpt


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
            if result_lines and result_lines[-1].strip() != '' and not re.match(r'^\s*[*+-]\s',
                                                                                result_lines[-1]) and not re.match(
                r'^\s*\d+\.\s', result_lines[-1]):
                result_lines.append('')
            result_lines.append(line)
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)


def transport_factory(proxy: httpx.Proxy | None = None) -> httpx.BaseTransport:
    return CurlTransport(  # or AsyncCurlTransport
        proxy=proxy,
        verify=False,  # and other custom options
    )


def make_request(content, session_id=None, conversation_history=None):
    # Use provided content for the request
    if conversation_history is None:
        conversation_history = []

    # Cookies as string
    cookies = "ph_phc_LG7IJbVJqBsk584rbcKca0D5lV2vHguiijDrVji7yDM_posthog=%7B%22distinct_id%22%3A%22e1bd59cf-aa49-4267-917d-4977c1c57a2a%22%2C%22%24sesid%22%3A%5B1756881790185%2C%2201990e15-4410-7adf-b225-04ef81809e7e%22%2C1756877898768%5D%2C%22%24epp%22%3Atrue%2C%22%24initial_person_info%22%3A%7B%22r%22%3A%22https%3A%2F%2Flmarena.ai%2Fc%2Fbb92e9de-bb63-4bb5-9b6b-554e602280bd%3F__cf_chl_tk%3Dis0RibDM0Clb0Um24oti2SwqiFf2REt9LFawvsfRn0c-1756177700-1.0.1.1-Qra4mogOwtB1FluvkW5NQUis47MfEJdtGG28AoZwn38%22%2C%22u%22%3A%22https%3A%2F%2Flmarena.ai%2Fc%2Fbb92e9de-bb63-4bb5-9b6b-554e602280bd%22%7D%7D; _ga_L5C4D55WJJ=GS2.1.s1756880113$o14$g1$t1756881763$j60$l0$h0; _ga=GA1.1.1915898.1756177709; arena-auth-prod-v1=; arena-auth-prod-v1-code-verifier=; arena-auth-prod-v1.0=base64-eyJhY2Nlc3NfdG9rZW4iOiJleUpoYkdjaU9pSklVekkxTmlJc0ltdHBaQ0k2SWtOVFQwNHhkM05uU0hkRlNFTkNNbGNpTENKMGVYQWlPaUpLVjFRaWZRLmV5SnBjM01pT2lKb2RIUndjem92TDJoMWIyZDZiMlZ4ZW1OeVpIWnJkM1IyYjJScExuTjFjR0ZpWVhObExtTnZMMkYxZEdndmRqRWlMQ0p6ZFdJaU9pSTFNVE00T1dZek1pMHlNall5TFRRMk9EWXRPVGswTnkxbE1qbGtZV0l3WVdVd09Ua2lMQ0poZFdRaU9pSmhkWFJvWlc1MGFXTmhkR1ZrSWl3aVpYaHdJam94TnpVMk9EZzFNelUwTENKcFlYUWlPakUzTlRZNE9ERTNOVFFzSW1WdFlXbHNJam9pYldGdVoyOHlNVEEzTURsQVoyMWhhV3d1WTI5dElpd2ljR2h2Ym1VaU9pSWlMQ0poY0hCZmJXVjBZV1JoZEdFaU9uc2ljSEp2ZG1sa1pYSWlPaUpuYjI5bmJHVWlMQ0p3Y205MmFXUmxjbk1pT2xzaVoyOXZaMnhsSWwxOUxDSjFjMlZ5WDIxbGRHRmtZWFJoSWpwN0ltRjJZWFJoY2w5MWNtd2lPaUpvZEhSd2N6b3ZMMnhvTXk1bmIyOW5iR1YxYzJWeVkyOXVkR1Z1ZEM1amIyMHZZUzlCUTJjNGIyTkpXRmc0VkdoRVlURklaM2hSUkhabE1scHZkVlZhTFhSWVdVWjRZVEJFZWtnMWVGTlJlbXN5UmxJeU1HaDVkMEU5Y3prMkxXTWlMQ0psYldGcGJDSTZJbTFoYm1kdk1qRXdOekE1UUdkdFlXbHNMbU52YlNJc0ltVnRZV2xzWDNabGNtbG1hV1ZrSWpwMGNuVmxMQ0ptZFd4c1gyNWhiV1VpT2lKdFlXNW5ieUlzSW1sa0lqb2laVEZpWkRVNVkyWXRZV0UwT1MwME1qWTNMVGt4TjJRdE5EazNOMk14WXpVM1lUSmhJaXdpYVhOeklqb2lhSFIwY0hNNkx5OWhZMk52ZFc1MGN5NW5iMjluYkdVdVkyOXRJaXdpYkdGemRGOXNhVzVyWldSZmMzVndZV0poYzJWZmRYTmxjbDlwWkNJNkltWTJaalV3T1dObUxXRm1aV0l0TkRSaU55MDVOVE13TFRZMFl6WXpZMkZsWVdWaE1TSXNJbTVoYldVaU9pSnRZVzVuYnlJc0luQm9iMjVsWDNabGNtbG1hV1ZrSWpwbVlXeHpaU3dpY0dsamRIVnlaU0k2SW1oMGRIQnpPaTh2YkdnekxtZHZiMmRzWlhWelpYSmpiMjUwWlc1MExtTnZiUzloTDBGRFp6aHZZMGxZV0RoVWFFUmhNVWhuZUZGRWRtVXlXbTkxVlZvdGRGaFpSbmhoTUVSNlNEVjRVMUY2YXpKR1VqSXdhSGwzUVQxek9UWXRZeUlzSW5CeWIzWnBaR1Z5WDJsa0lqb2lNVEExTWpVNU16RTBOemMyTURNeU1qRTVNRFUwSWl3aWMzVmlJam9pTVRBMU1qVTVNekUwTnpjMk1ETXlNakU1TURVMEluMHNJbkp2YkdVaU9pSmhkWFJvWlc1MGFXTmhkR1ZrSWl3aVlXRnNJam9pWVdGc01TSXNJbUZ0Y2lJNlczc2liV1YwYUc5a0lqb2liMkYxZEdnaUxDSjBhVzFsYzNSaGJYQWlPakUzTlRZNE9ERTNOVFI5WFN3aWMyVnpjMmx2Ymw5cFpDSTZJalZoTWpCaVpqTXlMVGM1T0RFdE5EUXdOQzA1WVRFM0xUVmpNalF3TVdRelpURmlPQ0lzSW1selgyRnViMjU1Ylc5MWN5STZabUZzYzJWOS5jMlE0aUxBYnViRFp1NzRIS1drc0dEXzNpT2FaVFRoMW44bDRUem5JODRNIiwidG9rZW5fdHlwZSI6ImJlYXJlciIsImV4cGlyZXNfaW4iOjM2MDAsImV4cGlyZXNfYXQiOjE3NTY4ODUzNTQsInJlZnJlc2hfdG9rZW4iOiJxNDYzdmtkZTVucjIiLCJ1c2VyIjp7ImlkIjoiNTEzODlmMzItMjI2Mi00Njg2LTk5NDctZTI5ZGFiMGFlMDk5IiwiYXVkIjoiYXV0aGVudGljYXRlZCIsInJvbGUiOiJhdXRoZW50aWNhdGVkIiwiZW1haWwiOiJtYW5nbzIxMDcwOUBnbWFpbC5jb20iLCJlbWFpbF9jb25maXJtZWRfYXQiOiIyMDI1LTA5LTAzVDAxOjUxOjA2LjYxMDk1NFoiLCJwaG9uZSI6IiIsImNvbmZpcm1lZF9hdCI6IjIwMjUtMDktMDNUMDE6NTE6MDYuNjEwOTU0WiIsImxhc3Rfc2lnbl9pbl9hdCI6IjIwMjUtMDktMDNUMDY6NDI6MzQuOTA3ODMyWiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6Imdvb2dsZSIsInByb3ZpZGVycyI6WyJnb29nbGUiXX0sInVzZXJfbWV0YWRhdGEiOnsiYXZhdGFyX3VybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0lYWDhUaERhMUhneFFEdmUyWm91VVotdFhZRnhhMER6SDV4U1F6azJGUjIwaHl3QT1zOTYtYyIsImVtYWlsIjoibWFuZ28yMTA3MDlAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZ1bGxfbmFtZSI6Im1hbmdvIiwiaWQiOiJlMWJkNTljZi1hYTQ5LTQyNjctOTE3ZC00OTc3YzFjNTdhMmEiLCJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJsYXN0X2xpbmtlZF9zdXBhYmFzZV91c2VyX2lkIjoiNjg3MWFhZDYtMjA3Mi00MGY2LTk1OGItM2JmNGViMThiMDMwIiwibmFtZSI6Im1hbmdvIiwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jSVhYOFRoRGExSGd4UUR2ZTJab3VVWi10WFlGeGEwRHpINXhTUXprMkZSM; arena-auth-prod-v1.1=jBoeXdBPXM5Ni1jIiwicHJvdmlkZXJfaWQiOiIxMDUyNTkzMTQ3NzYwMzIyMTkwNTQiLCJzdWIiOiIxMDUyNTkzMTQ3NzYwMzIyMTkwNTQifSwiaWRlbnRpdGllcyI6W3siaWRlbnRpdHlfaWQiOiI2Yjk1N2E4MS0zMmQ4LTRkY2QtOGY5Mi0zODFjNGYwNDVkZjgiLCJpZCI6IjEwNTI1OTMxNDc3NjAzMjIxOTA1NCIsInVzZXJfaWQiOiI1MTM4OWYzMi0yMjYyLTQ2ODYtOTk0Ny1lMjlkYWIwYWUwOTkiLCJpZGVudGl0eV9kYXRhIjp7ImF2YXRhcl91cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NJWFg4VGhEYTFIZ3hRRHZlMlpvdVVaLXRYWUZ4YTBEekg1eFNRemsyRlIyMGh5d0E9czk2LWMiLCJlbWFpbCI6Im1hbmdvMjEwNzA5QGdtYWlsLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJmdWxsX25hbWUiOiJtYW5nbyIsImlzcyI6Imh0dHBzOi8vYWNjb3VudHMuZ29vZ2xlLmNvbSIsIm5hbWUiOiJtYW5nbyIsInBob25lX3ZlcmlmaWVkIjpmYWxzZSwicGljdHVyZSI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0lYWDhUaERhMUhneFFEdmUyWm91VVotdFhZRnhhMER6SDV4U1F6azJGUjIwaHl3QT1zOTYtYyIsInByb3ZpZGVyX2lkIjoiMTA1MjU5MzE0Nzc2MDMyMjE5MDU0Iiwic3ViIjoiMTA1MjU5MzE0Nzc2MDMyMjE5MDU0In0sInByb3ZpZGVyIjoiZ29vZ2xlIiwibGFzdF9zaWduX2luX2F0IjoiMjAyNS0wOS0wM1QwMTo1MTowNi42MDY0MTFaIiwiY3JlYXRlZF9hdCI6IjIwMjUtMDktMDNUMDE6NTE6MDYuNjA2NDU3WiIsInVwZGF0ZWRfYXQiOiIyMDI1LTA5LTAzVDA2OjQyOjM0LjU2NTY4M1oiLCJlbWFpbCI6Im1hbmdvMjEwNzA5QGdtYWlsLmNvbSJ9XSwiY3JlYXRlZF9hdCI6IjIwMjUtMDktMDNUMDE6NTE6MDYuNjA0OTA3WiIsInVwZGF0ZWRfYXQiOiIyMDI1LTA5LTAzVDA2OjQyOjM0Ljk1NDE3NFoiLCJpc19hbm9ueW1vdXMiOmZhbHNlfSwicHJvdmlkZXJfdG9rZW4iOiJ5YTI5LkEwQVMzSDZOeEpMb1ZZcUVqTGJ3dkwxZXlpOVVQSUg5SEFweFptRWF4ZG5Lck53YjB3NE5oNW0tYUNTOThJcTY3aU1MbEZFdHY5Uk1FME80Q1JIXzdUTUlwb2FQR2RWeXg0QWxnb3ZRekpzeWxBQ2szVFZUV0RWOWZYc1FmVDhnN0I4YmNRRWhFOFFjSjdjbzI1UWRhVnZ1T25vNEV6UGkwZ2QtOTYtTHFLVnpBZXhPYmFpNF8wVjFjLWlLLU5ETDVubG0zUzl1b2FDZ1lLQVZnU0FSWVNGUUhHWDJNaVVzTjVxYUhmVGs4OXg5R2xwcV9nZncwMjA2IiwicHJvdmlkZXJfcmVmcmVzaF90b2tlbiI6IjEvLzA2WW1FaEpCSng3OHlDZ1lJQVJBQUdBWVNOd0YtTDlJcmJIZklHTjA2Rmp0RGhUeTlpNENZYTRZYUlZbEF0cWZQbThkZHhrYTYxVl9UU0RUV213M0t4Um5KR1UtaWV2TUFiTjgifQ; cf_clearance=8tEJTjoEgLkrbKSjlWgo5g5thZh1V8V.Ii_9oO2vuaI-1756881745-1.2.1.1-pkAYebQh4T1XOeJ0m9kY0zMJwoHYXpwAV1cojwe36DhRfs38maKDq1wBGg3XPQI4YPw617nOhtHmPx9Ss.5RRc6AFGgBj6ehmKvRdLHTDqsqrSLOU_4hTIzqwMbNbfhZjXTYEoVHO3kkiRdE7eJKgQyHaMQXn9Yutgl8x.mEnzEjSCb1zfBlRJrbNqBTK6eA_UzmT0ioKQKm3ZNx_Ignhf5GyZab5N3glf0iTnM6729oEiyiEbj_Ywu7TYFhsntT; __cf_bm=6M7Xyq.qDMFramSHdyMBLC3s_6Wtf9QXVlxKicbwePM-1756881711-1.0.1.1-5KSKmn750dlFb85RFXdcZBNoFu2iqapDdi1Ed6k4zk5KCnwz6Q8CZJZghGWeAhr2syOpIF9Ifb3LA3IFWkdpXeG.2h445KmvrjQiWb7ladU; sidebar_state=true"

    # Headers
    headers = {
        "Host": "lmarena.ai",
        "Connection": "keep-alive",
        "sec-ch-ua-full-version-list": '"Not;A=Brand";v="99.0.0.0", "Google Chrome";v="139.0.7258.155", "Chromium";v="139.0.7258.155"',
        "sec-ch-ua-platform": '"macOS"',
        "sec-ch-ua": '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
        "sec-ch-ua-bitness": '"64"',
        "sec-ch-ua-model": '""',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-arch": '"arm"',
        "sec-ch-ua-full-version": '"139.0.7258.155"',
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "sec-ch-ua-platform-version": '"26.0.0"',
        "Accept": "*/*",
        "Origin": "https://lmarena.ai",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://lmarena.ai/?mode=direct",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cache-Control": "no-cache",
        "Postman-Token": "52271d3c-9a6b-4faa-9f3a-719cbb660ad6",
        "Content-Type": "text/plain;charset=UTF-8",
        "Cookie": cookies
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
        "modelAId": "983bc566-b783-4d28-b24c-3c8b08eb1086",
        "userMessageId": user_message_id,
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


def save_conversation_history(conversation_history, session_id):
    """Save conversation history to chat directory"""
    if not conversation_history:
        return

    # Create chat directory if it doesn't exist
    chat_dir = "chat"
    os.makedirs(chat_dir, exist_ok=True)

    # Generate filename with timestamp
    filename = f"{session_id}.json"
    filepath = os.path.join(chat_dir, filename)

    # Prepare conversation data
    chat_data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "conversation_history": conversation_history,
        "message_count": len(conversation_history)
    }

    # Save to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        print(f"对话已保存到: {filename}")
    except Exception as e:
        print(f"保存对话失败: {e}")


def handle_special_commands(user_input, conversation_history, session_id):
    """Handle special commands like quit, exit, clear"""
    if user_input.lower() in ['quit', 'exit', '退出']:
        # Save conversation before exit
        if conversation_history and session_id:
            save_conversation_history(conversation_history, session_id)
        print("再见！")
        return "exit"
    elif user_input.lower() in ['clear', '清空']:
        # Save conversation before clearing
        if conversation_history and session_id:
            save_conversation_history(conversation_history, session_id)
        print("对话历史已清空")
        return "clear"
    return None


def conversation_loop(session_id=None, conversation_history=None):
    """Main conversation loop"""
    if session_id is None:
        session_id = None
    if conversation_history is None:
        conversation_history = []

    while True:
        try:
            user_input = input("你: ").strip()
            if not user_input:
                continue
            command_result = handle_special_commands(user_input, conversation_history, session_id)
            if command_result == "exit":
                break
            elif command_result == "clear":
                session_id = None
                conversation_history = []
                continue
            print("\n助手: ", end='', flush=True)
            result = make_request(user_input, session_id, conversation_history)

            if result:
                session_id = result["session_id"]
                conversation_history = result["conversation_history"]
            else:
                print("请求失败，请重试")

        except KeyboardInterrupt:
            print("\n\n对话已中断")
            # Save conversation before exit
            if conversation_history and session_id:
                save_conversation_history(conversation_history, session_id)
            break
        except Exception as e:
            print(f"\n发生错误: {e}")


def start_conversation():
    """Start a multi-turn conversation session"""
    print("输入 'clear' 清空对话历史")
    print("-" * 50)

    conversation_loop()


if __name__ == "__main__":
    # Check if there's a command line argument for starting conversation with initial message
    if len(sys.argv) > 1:
        initial_content = sys.argv[1]
        print("输入 'clear' 清空对话历史")
        print("-" * 50)

        session_id = None
        conversation_history = []

        # Send initial message
        print("你: " + initial_content)
        print("\n助手: ", end='', flush=True)
        result = make_request(initial_content, session_id, conversation_history)

        if result:
            session_id = result["session_id"]
            conversation_history = result["conversation_history"]

        # Continue with interactive conversation
        conversation_loop(session_id, conversation_history)
    else:
        # Start multi-turn conversation
        start_conversation()
