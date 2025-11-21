from __future__ import annotations

import os
import os.path
from typing import List, Dict, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build



SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


ENV_CLIENT_ID = "GOOGLE_CLIENT_ID"
ENV_CLIENT_SECRET = "GOOGLE_CLIENT_SECRET"
ENV_PROJECT_ID = "GOOGLE_PROJECT_ID"



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



def fetch_latest_messages(service, max_results: int = 5) -> List[Dict[str, Any]]:
    
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
            .get(userId="me", id=summary["id"], format="metadata", metadataHeaders=["From", "Subject"])
            .execute()
        )
        messages.append(msg)

    return messages


def print_sender_and_subject(messages: List[Dict[str, Any]]) -> None:
    """
    Given a list of Gmail messages, print sender and subject for each
    """
    if not messages:
        print("No messages found.")
        return

    for msg in messages:
        headers = msg.get("payload", {}).get("headers", [])
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "(No subject)")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "(Unknown sender)")
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        print("-" * 60)



def main():
    print("Connecting to Gmail…")
    service = get_gmail_service()
    print("Connected. Fetching latest emails…")

    messages = fetch_latest_messages(service, max_results=5)
    print_sender_and_subject(messages)


if __name__ == "__main__":
    main()