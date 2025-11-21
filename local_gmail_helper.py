from __future__ import annotations

"""
Environment variables:
  GOOGLE_CLIENT_ID      (required)
  GOOGLE_CLIENT_SECRET  (required)
  GOOGLE_PROJECT_ID     (required)
  LLM_BASE_URL          (optional, default "http://127.0.0.1:11434")
  LLM_MODEL             (optional, default "llama3.1")
"""

import os
import os.path
import json
import argparse
import base64  # NEW
from typing import List, Dict, Any, Tuple
from email.utils import parseaddr

import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

ENV_CLIENT_ID = "GOOGLE_CLIENT_ID"
ENV_CLIENT_SECRET = "GOOGLE_CLIENT_SECRET"
ENV_PROJECT_ID = "GOOGLE_PROJECT_ID"

ENV_LLM_BASE_URL = "LLM_BASE_URL"
ENV_LLM_MODEL = "LLM_MODEL"

LLM_BASE_URL = os.environ.get(ENV_LLM_BASE_URL, "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get(ENV_LLM_MODEL, "llama3.1")


def get_gmail_service():
    creds: Credentials | None = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_id = os.environ.get(ENV_CLIENT_ID)
            client_secret = os.environ.get(ENV_CLIENT_SECRET)
            project_id = os.environ.get(ENV_PROJECT_ID)

            if not client_id or not client_secret or not project_id:
                raise RuntimeError(
                    "Missing one of the required environment variables: "
                    f"{ENV_CLIENT_ID}, {ENV_CLIENT_SECRET}, {ENV_PROJECT_ID}"
                )

            client_config = {
                "installed": {
                    "client_id": client_id,
                    "project_id": project_id,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                    "client_secret": client_secret,
                    "redirect_uris": [
                        "urn:ietf:wg:oauth:2.0:oob",
                        "http://localhost",
                    ],
                }
            }

            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    service = build("gmail", "v1", credentials=creds)
    return service


def fetch_latest_messages(service, max_results: int = 50) -> List[Dict[str, Any]]:
    result = (
        service.users()
        .messages()
        .list(userId="me", maxResults=max_results, labelIds=["INBOX"])
        .execute()
    )
    message_summaries = result.get("messages", [])
    messages: List[Dict[str, Any]] = []

    for summary in message_summaries:
        msg = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=summary["id"],
                format="full",
            )
            .execute()
        )
        messages.append(msg)

    return messages


def extract_body_excerpt(payload: Dict[str, Any], max_chars: int = 400) -> str:
    """
    Extract a short text/plain excerpt from the email body.
    Keeps it small so prompts stay efficient for the local LLM.
    """
    texts: List[str] = []

    def decode_body(data: str) -> str:
        if not data:
            return ""
        padding = 4 - (len(data) % 4)
        if padding and padding != 4:
            data += "=" * padding
        try:
            decoded_bytes = base64.urlsafe_b64decode(data)
            return decoded_bytes.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def walk(part: Dict[str, Any]):
        if len(" ".join(texts)) >= max_chars:
            return

        mime_type = part.get("mimeType", "")
        body = part.get("body", {})

        if mime_type.startswith("multipart/") and "parts" in part:
            for sub in part["parts"]:
                walk(sub)
                if len(" ".join(texts)) >= max_chars:
                    break
        else:
            if mime_type == "text/plain":
                data = body.get("data")
                if data:
                    text = decode_body(data)
                    if text.strip():
                        texts.append(text)

    if not payload:
        return ""

    walk(payload)

    if not texts:
        return ""

    combined = " ".join(texts)
    combined = " ".join(combined.split())

    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "..."

    return combined


def extract_email_info(msg: Dict[str, Any], max_body_chars: int = 400) -> Dict[str, str]:
    headers = msg.get("payload", {}).get("headers", [])

    def get_header(name: str) -> str:
        return next(
            (h["value"] for h in headers if h["name"].lower() == name.lower()),
            "",
        )

    from_raw = get_header("From")
    to_raw = get_header("To")
    subject = get_header("Subject")
    snippet = msg.get("snippet", "")

    sender_name, sender_address = parseaddr(from_raw)
    _, to_address = parseaddr(to_raw)

    
    body_excerpt = extract_body_excerpt(msg.get("payload", {}), max_chars=max_body_chars)
    if not body_excerpt:
        body_excerpt = snippet or ""

    return {
        "sender_name": sender_name or sender_address or "(unknown)",
        "sender_address": sender_address or "(unknown)",
        "to_address": to_address or "(unknown)",
        "subject": subject or "(no subject)",
        "snippet": snippet or "",
        "body_excerpt": body_excerpt,  # NEW
    }


def build_email_prompt(info: Dict[str, str]) -> str:
    body_excerpt = info.get("body_excerpt", "").strip()

    prompt = f"""
You are an assistant that classifies emails.

Given the email details below, you must answer in valid JSON only with the keys:
  - "category": one of ["Work","School","Shopping","Finance","Personal","Social","Spam","Other"]
  - "priority": one of ["Urgent","Important","Normal","Low"]
  - "requires_response": true or false

Do NOT include any explanation, just the JSON.

Email details:
- Sender name: {info["sender_name"]}
- Sender address: {info["sender_address"]}
- Recipient address: {info["to_address"]}
- Subject: {info["subject"]}
- Snippet: {info["snippet"]}
"""

    if body_excerpt:
        prompt += f"- Body excerpt (truncated): {body_excerpt}\n"

    return prompt.strip()


def call_local_llm(prompt: str) -> Dict[str, Any]:
    base = LLM_BASE_URL.rstrip("/")
    url = f"{base}/api/chat"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON-only email classifier.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(url, json=payload, timeout=(5, 120))
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Timed out talking to Ollama at {url}. "
            f"Make sure the Ollama server is running and the model '{LLM_MODEL}' "
            f"is loaded. Try running 'ollama run {LLM_MODEL}' in a terminal first "
            f"and check that it responds quickly."
        )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Error calling Ollama at {url}: {e}") from e

    data = resp.json()
    content = data.get("message", {}).get("content", "")
    content_str = (content or "").strip()

    if content_str.startswith("```"):
        content_str = content_str.strip("`")
        if content_str.lower().startswith("json"):
            content_str = content_str[4:].strip()

    try:
        return json.loads(content_str)
    except json.JSONDecodeError:
        return {"raw_response": content_str}


def analyze_emails(max_emails: int = 50):
    print(f"LLM_BASE_URL = {LLM_BASE_URL}")
    print(f"LLM_MODEL    = {LLM_MODEL}")
    print(f"Connecting to Gmail and fetching last {max_emails} emails...")
    service = get_gmail_service()
    messages = fetch_latest_messages(service, max_results=max_emails)

    print(f"Fetched {len(messages)} messages.")
    print("=" * 80)

    results: List[Tuple[Dict[str, str], Dict[str, Any]]] = []

    for i, msg in enumerate(messages, start=1):
        info = extract_email_info(msg, max_body_chars=400)  
        prompt = build_email_prompt(info)

        print(f"[{i}/{len(messages)}] Asking LLM about email: {info['subject']!r}")
        llm_result = call_local_llm(prompt)
        results.append((info, llm_result))

    print("\n=== Classification results ===\n")
    for info, llm_result in results:
        category = llm_result.get("category", "Unknown")
        priority = llm_result.get("priority", "Unknown")
        requires_response = llm_result.get(
            "requires_response",
            llm_result.get("requires_reply", False),
        )

        print(f"Subject : {info['subject']}")
        print(f"From    : {info['sender_name']} <{info['sender_address']}>")
        print(f"Category: {category}")
        print(f"Priority: {priority}")
        print(f"Needs reply? {requires_response}")
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Gmail emails with a local LLM."
    )
    parser.add_argument(
        "--max-emails",
        type=int,
        default=50,
        help="Number of latest emails to analyze (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_emails(max_emails=args.max_emails)


if __name__ == "__main__":
    main()
