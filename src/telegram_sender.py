# -*- coding: utf-8 -*-

import requests
import os

def send_telegram_message(message, silent=False):
    """Sends a message to Telegram using MarkdownV2 or HTML."""
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        print("Error: Telegram credentials (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) not set.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    # Telegram MarkdownV2 requires escaping special characters
    # HTML is generally easier to work with for bold/links
    parse_mode = 'MarkdownV2'
    
    # Simple Markdown to MDV2/HTML conversion & escaping
    def escape_markdown(text):
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        return ''.join(['\\' + char if char in escape_chars else char for char in text])

    # Let's try HTML as it's often more forgiving
    parse_mode = 'HTML'
    # Convert simple markdown to HTML (Handle with care or use a library)
    message = message.replace('**', '<b>').replace('</b><b>', '</b>').replace('__', '<i>').replace('</i><i>', '</i>')
    message = message.replace('🚨', '<b>🚨').replace('</b>\n\n', '</b>\n\n') # Ensure bold titles

    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': parse_mode,
        'disable_web_page_preview': True,
        'disable_notification': silent
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            return True
        else:
            print(f"Error sending message to Telegram ({response.status_code}): {response.text}")
            # Try sending as plain text if formatting fails
            payload['parse_mode'] = None
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
