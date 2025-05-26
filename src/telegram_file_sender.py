# -*- coding: utf-8 -*-

import requests
import os # Ensure os is imported
import sys
import glob

def send_telegram_file(file_path, caption=""):
    """Sends a specific file to the configured Telegram chat."""
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        print("Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables not set.")
        return False

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"

    if not caption:
        caption = f"📊 Backtest Report: {os.path.basename(file_path)}"

    print(f"Sending {file_path} to Telegram...")

    try:
        with open(file_path, 'rb') as f:
            files = {'document': (os.path.basename(file_path), f)}
            data = {'chat_id': chat_id, 'caption': caption}
            response = requests.post(url, files=files, data=data, timeout=60)

            if response.status_code == 200:
                print(f"File {file_path} sent successfully to Telegram.")
                return True
            else:
                print(f"Error sending file to Telegram ({response.status_code}): {response.text}")
                return False
    except Exception as e:
        print(f"An exception occurred while sending file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.telegram_file_sender <file_path_or_pattern>")
        sys.exit(1)

    file_pattern = sys.argv[1]
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"No files found matching pattern: {file_pattern}")
        sys.exit(1)

    print(f"Found {len(file_list)} file(s) to send.")

    all_sent = True
    for f_path in file_list:
        if not send_telegram_file(f_path):
            all_sent = False

    if not all_sent:
        print("One or more files failed to send.")
        sys.exit(1)
    else:
        print("All files sent successfully.")
        sys.exit(0)
