# -*- coding: utf-8 -*-

import requests
import os

def send_telegram_message(message, silent=False):
    """Sends a message to Telegram using HTML."""
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        print("Error: Telegram credentials (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) not set.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML', # Always try HTML first
        'disable_web_page_preview': True, # Keep True for cleaner messages
        'disable_notification': silent
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            return True
        else:
            print(f"Error sending message to Telegram ({response.status_code}): {response.text}")
            # Fallback: Remove parse_mode for plain text
            payload.pop('parse_mode', None) # Remove the key
            print("Trying to send as plain text...")
            response_plain = requests.post(url, json=payload, timeout=15)
            if response_plain.status_code == 200:
                print("Sent as plain text after format error.")
                return True
            else:
                print(f"Plain text send also failed ({response_plain.status_code}): {response_plain.text}")
                return False
    except Exception as e:
        print(f"Exception sending message to Telegram: {e}")
        return False

