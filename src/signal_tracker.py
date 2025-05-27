# -*- coding: utf-8 -*-

import json
import os
import requests
import argparse
from datetime import datetime, timedelta
import pytz
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from filelock import FileLock
from config import SIGNALS_FILE, KUCOIN_BASE_URL, KUCOIN_KLINE_ENDPOINT, KUCOIN_TICKER_ENDPOINT
from telegram_sender import send_telegram_message

def load_signals():
    """Load signals from JSON file with proper timezone handling and validation."""
    lock = FileLock(f"{SIGNALS_FILE}.lock")
    try:
        with lock:
            if not os.path.exists(SIGNALS_FILE):
                print(f"No signals file found at {SIGNALS_FILE}. Creating an empty one.")
                os.makedirs(os.path.dirname(SIGNALS_FILE), exist_ok=True)
                with open(SIGNALS_FILE, 'w') as f:
                    f.write("[]")
                return []

            with open(SIGNALS_FILE, 'r') as f:
                content = f.read()
                if not content.strip():
                    print(f"Signals file {SIGNALS_FILE} is empty.")
                    return []
                signals = json.loads(content)
                print(f"Loaded {len(signals)} signals from {SIGNALS_FILE}")

                tehran_tz = pytz.timezone('Asia/Tehran')
                valid_signals = []
                for signal in signals:
                    try:
                        # Ensure basic fields exist
                        if not all(k in signal for k in ['symbol', 'type', 'current_price', 'target_price', 'stop_loss', 'created_at']):
                            print(f"Skipping invalid signal (missing fields): {signal.get('symbol', 'unknown')}")
                            continue

                        # Ensure valid status
                        if 'status' not in signal or signal['status'] not in ['active', 'target_reached', 'stop_loss']:
                            print(f"Fixing invalid status for {signal.get('symbol', 'unknown')}")
                            signal['status'] = 'active'

                        # Ensure created_at is timezone-aware ISO format
                        created_at_str = signal['created_at']
                        if 'T' in created_at_str and ('+' in created_at_str or 'Z' in created_at_str):
                             created_at = datetime.fromisoformat(created_at_str)
                        else: # Try to parse older formats or non-iso
                             try:
                                 created_at = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                                 created_at = tehran_tz.localize(created_at)
                             except ValueError:
                                 print(f"Could not parse created_at for {signal['symbol']}: {created_at_str}. Skipping.")
                                 continue
                        signal['created_at'] = created_at.isoformat()

                        # Handle closed_at if present
                        if 'closed_at' in signal and signal['closed_at']:
                            closed_at_str = signal['closed_at']
                            if 'T' in closed_at_str and ('+' in closed_at_str or 'Z' in closed_at_str):
                                closed_at = datetime.fromisoformat(closed_at_str)
                            else:
                                try:
                                    closed_at = datetime.strptime(closed_at_str, "%Y-%m-%d %H:%M:%S")
                                    closed_at = tehran_tz.localize(closed_at)
                                except ValueError:
                                    closed_at = None # In case of parsing error
                            signal['closed_at'] = closed_at.isoformat() if closed_at else None

                        valid_signals.append(signal)
                    except Exception as e:
                        print(f"Error processing a signal ({signal.get('symbol', 'unknown')}): {e}. Skipping.")

                return valid_signals
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {SIGNALS_FILE}: {e}")
        # Optionally: handle corrupted file (backup/rename and start fresh)
        return []
    except Exception as e:
        print(f"Error loading signals: {e}")
        return []

def save_signals(signals):
    """Save signals to JSON file with proper timezone handling and file lock."""
    lock = FileLock(f"{SIGNALS_FILE}.lock")
    try:
        with lock:
            os.makedirs(os.path.dirname(SIGNALS_FILE), exist_ok=True)
            with open(SIGNALS_FILE, 'w') as f:
                json.dump(signals, f, indent=4) # Use indent 4 for better readability
            print(f"Saved {len(signals)} signals to {SIGNALS_FILE}")
    except Exception as e:
        print(f"Error saving signals: {e}")
        send_telegram_message(f"❌ Error saving signals: {e}")

def save_signal(signal):
    """Save a single signal with proper timezone handling."""
    tehran_tz = pytz.timezone('Asia/Tehran')
    if 'entry_price' not in signal or not signal['entry_price']:
        signal['entry_price'] = signal.get('current_price')
    signal['status'] = 'active'
    if 'created_at' not in signal or not signal['created_at']:
         signal['created_at'] = datetime.now(tehran_tz).isoformat()
    elif isinstance(signal['created_at'], str) and 'T' not in signal['created_at']:
         # Ensure it is in ISO format
         try:
            dt_obj = datetime.strptime(signal['created_at'], "%Y-%m-%d %H:%M:%S")
            signal['created_at'] = tehran_tz.localize(dt_obj).isoformat()
         except ValueError:
             signal['created_at'] = datetime.now(tehran_tz).isoformat() # Fallback

    signals = load_signals()
    signals.append(signal)
    save_signals(signals)
    print(f"Signal saved: {signal['symbol']} {signal['type']}")

def fetch_kline_data(symbol, start_time, end_time, interval="30min"):
    """Fetch kline data from KuCoin for a specific time range"""
    url = f"{KUCOIN_BASE_URL}{KUCOIN_KLINE_ENDPOINT}"
    params = {
        "symbol": symbol,
        "type": interval,
        "startAt": int(start_time.timestamp()),
        "endAt": int(end_time.timestamp())
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data.get('data'):
            print(f"No kline data for {symbol} from {start_time} to {end_time}: {data}")
            return None
        df = pd.DataFrame(data['data'], columns=[
            "timestamp", "open", "close", "high", "low", "volume", "turnover"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df = df.astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize('UTC').dt.tz_convert('Asia/Tehran')
        df = df.iloc[::-1].reset_index(drop=True)
        print(f"Received {len(df)} candles for {symbol} from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
        return df
    except Exception as e:
        print(f"Error fetching kline data for {symbol}: {e}")
        return None

def get_current_price(symbol):
    """Fetch current price from KuCoin"""
    url = f"{KUCOIN_BASE_URL}{KUCOIN_TICKER_ENDPOINT}"
    params = {"symbol": symbol}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        price = data.get('data', {}).get('price')
        if price:
            print(f"Current price for {symbol}: {price}")
            return float(price)
        print(f"No price data for {symbol}: {data}")
        return None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

def calculate_profit_loss(signal, close_price):
    """Calculate profit/loss percentage"""
    try:
        entry_price_str = signal.get('entry_price', signal['current_price'])
        if entry_price_str is None: return None
        entry_price = float(entry_price_str)
        close_price = float(close_price)
        if entry_price == 0: return None # Avoid division by zero
        if signal['type'] == 'BUY':
            return ((close_price - entry_price) / entry_price) * 100
        else:  # SELL
            return ((entry_price - close_price) / entry_price) * 100
    except (ValueError, TypeError, KeyError) as e:
        print(f"Error calculating profit/loss for {signal.get('symbol', 'N/A')}: {e}")
        return None

def calculate_duration(created_at_str, closed_at_str):
    """Calculate signal duration in hours"""
    tehran_tz = pytz.timezone('Asia/Tehran')
    try:
        created = datetime.fromisoformat(created_at_str).astimezone(tehran_tz)
        
        if closed_at_str:
            closed = datetime.fromisoformat(closed_at_str).astimezone(tehran_tz)
        else:
            closed = datetime.now(tehran_tz)
        
        duration_hours = (closed - created).total_seconds() / 3600
        return duration_hours
    except (ValueError, TypeError) as e:
        print(f"Error calculating duration for {created_at_str}: {e}")
        return None

def check_signal_hit(signal, df):
    """Check if signal hit target or stop-loss based on kline data"""
    try:
        target_price = float(signal['target_price'])
        stop_loss = float(signal['stop_loss'])
        signal_type = signal['type']
        created_at = datetime.fromisoformat(signal['created_at']).astimezone(pytz.timezone('Asia/Tehran'))

        for _, row in df.iterrows():
            candle_time = row['timestamp']
            if candle_time <= created_at:
                continue  # Skip candles before or at signal creation

            high = row['high']
            low = row['low']
            
            if signal_type == 'BUY':
                if high >= target_price:
                    return 'target_reached', str(target_price), candle_time.isoformat()
                if low <= stop_loss:
                    return 'stop_loss', str(stop_loss), candle_time.isoformat()
            elif signal_type == 'SELL':
                if low <= target_price:
                    return 'target_reached', str(target_price), candle_time.isoformat()
                if high >= stop_loss:
                    return 'stop_loss', str(stop_loss), candle_time.isoformat()
        return None, None, None
    except Exception as e:
        print(f"Error checking signal hit for {signal['symbol']}: {e}")
        return None, None, None

def update_signal_status():
    """Update signal statuses by checking historical kline data."""
    signals = load_signals()
    if not signals:
        print("No signals to update")
        return

    updated = False
    tehran_tz = pytz.timezone('Asia/Tehran')
    now = datetime.now(tehran_tz)
    
    for signal in signals:
        if signal['status'] != 'active':
            # print(f"Skipping {signal['symbol']}: Already {signal['status']}")
            continue

        try:
            created_at = datetime.fromisoformat(signal['created_at']).astimezone(tehran_tz)
            # Fetch kline data from signal creation up to now + a buffer
            df = fetch_kline_data(signal['symbol'], created_at - timedelta(minutes=30), now + timedelta(minutes=30), interval="30min")
            if df is None or df.empty:
                print(f"Skipping update for {signal['symbol']} due to missing kline data")
                continue

            status, closed_price, closed_at_iso = check_signal_hit(signal, df)
            if status:
                signal['status'] = status
                signal['closed_price'] = closed_price
                signal['closed_at'] = closed_at_iso
                updated = True
                print(f"Updated {signal['symbol']}: {status} at {closed_price} on {closed_at_iso}")
                send_telegram_message(
                    f"📢 **Signal Update: {signal['symbol']}**\n"
                    f"**Status:** {status.replace('_', ' ').title()}\n"
                    f"**Closed Price:** {closed_price}\n"
                    f"**Time:** {datetime.fromisoformat(closed_at_iso).strftime('%Y-%m-%d %H:%M:%S')}"
                )
            # Optional: Add a timeout check - close active signals older than X days
            # elif (now - created_at).days > 7:
            #     signal['status'] = 'timed_out'
            #     signal['closed_price'] = str(get_current_price(signal['symbol']))
            #     signal['closed_at'] = now.isoformat()
            #     updated = True
            #     print(f"Updated {signal['symbol']}: Timed Out")

        except Exception as e:
            print(f"Error updating {signal['symbol']}: {e}")

    if updated:
        save_signals(signals)
        print("Signals updated successfully")
    else:
        print("No signals were updated")

def send_telegram_file(file_path):
    """Send file to Telegram"""
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')

    if not bot_token or not chat_id:
        print("Error: Telegram credentials not set")
        return False

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    try:
        with open(file_path, 'rb') as f:
            files = {'document': (os.path.basename(file_path), f)}
            data = {
                'chat_id': chat_id,
                'caption': '📊 Signals Report'
            }
            response = requests.post(url, files=files, data=data, timeout=30) # Increased timeout
            if response.status_code == 200:
                print(f"File {file_path} sent to Telegram")
                return True
            else:
                print(f"Error sending file to Telegram ({response.status_code}): {response.text}")
                return False
    except Exception as e:
        print(f"Error sending file to Telegram: {e}")
        return False

def generate_excel_report():
    """Generate Excel report with multiple sheets"""
    update_signal_status() # Ensure status is up-to-date before reporting
    signals = load_signals()
    tehran_tz = pytz.timezone('Asia/Tehran')
    now_str = datetime.now(tehran_tz).strftime("%Y%m%d_%H%M%S")
    os.makedirs('data', exist_ok=True)
    output_file = f"data/signals_report_{now_str}.xlsx"

    all_signals_data = []
    active_signals_data = []

    for signal in signals:
        current_price = None
        if signal['status'] == 'active':
            current_price = get_current_price(signal['symbol'])
            time.sleep(0.3) # Avoid hitting API rate limits

        close_price_str = signal.get('closed_price')
        close_price = float(close_price_str) if close_price_str else current_price
        
        profit_loss = calculate_profit_loss(signal, close_price) if close_price is not None else None
        duration = calculate_duration(signal['created_at'], signal.get('closed_at'))
        
        entry_price_str = signal.get('entry_price', signal['current_price'])
        
        signal_row = {
            'Symbol': signal['symbol'],
            'Type': signal['type'],
            'Entry_Price': float(entry_price_str) if entry_price_str else None,
            'Target_Price': float(signal['target_price']),
            'Stop_Loss': float(signal['stop_loss']),
            'Created_At': datetime.fromisoformat(signal['created_at']).strftime('%Y-%m-%d %H:%M'),
            'Status': signal['status'],
            'Closed_Price': float(signal['closed_price']) if signal.get('closed_price') else None,
            'Closed_At': datetime.fromisoformat(signal['closed_at']).strftime('%Y-%m-%d %H:%M') if signal.get('closed_at') else None,
            'Profit_Loss_%': round(profit_loss, 2) if profit_loss is not None else None,
            'Duration_Hours': round(duration, 2) if duration is not None else None,
            'Reasons': signal.get('reasons', '').replace('✅ ', '').replace('\n', '; ')
        }
        all_signals_data.append(signal_row)
        
        if signal['status'] == 'active' and current_price is not None and signal_row['Entry_Price'] is not None:
            price_change = ((current_price - signal_row['Entry_Price']) / signal_row['Entry_Price']) * 100 if signal_row['Entry_Price'] != 0 else 0
            active_signals_data.append({
                'Symbol': signal['symbol'],
                'Type': signal['type'],
                'Entry_Price': signal_row['Entry_Price'],
                'Current_Price': current_price,
                'Price_Change_%': round(price_change, 2),
                'Created_At': signal_row['Created_At'],
                'Reasons': signal_row['Reasons']
            })

    # Calculate statistics
    closed_signals = [s for s in all_signals_data if s['Status'] in ['target_reached', 'stop_loss']]
    total_signals = len(all_signals_data)
    active_count = len(active_signals_data)
    target_reached = len([s for s in closed_signals if s['Status'] == 'target_reached'])
    stop_loss_signals = len([s for s in closed_signals if s['Status'] == 'stop_loss'])
    total_closed = target_reached + stop_loss_signals
    success_rate = (target_reached / total_closed * 100) if total_closed > 0 else 0
    
    profits = [s['Profit_Loss_%'] for s in closed_signals if s['Profit_Loss_%'] is not None]
    avg_profit = pd.Series(profits).mean() if profits else 0
    
    durations = [s['Duration_Hours'] for s in closed_signals if s['Duration_Hours'] is not None]
    avg_duration = pd.Series(durations).mean() if durations else 0

    stats_data = [
        {'Metric': 'Total Signals', 'Value': total_signals},
        {'Metric': 'Active Signals', 'Value': active_count},
        {'Metric': 'Target Reached', 'Value': target_reached},
        {'Metric': 'Stop Loss Hit', 'Value': stop_loss_signals},
        {'Metric': 'Total Closed', 'Value': total_closed},
        {'Metric': 'Success Rate (%)', 'Value': round(success_rate, 2)},
        {'Metric': 'Average Profit/Loss (%)', 'Value': round(avg_profit, 2) if pd.notna(avg_profit) else None},
        {'Metric': 'Average Duration (Hours)', 'Value': round(avg_duration, 2) if pd.notna(avg_duration) else None}
    ]

    # Create Excel file
    wb = Workbook()
    
    # Sheet 1: All Signals
    ws1 = wb.active
    ws1.title = "All Signals"
    headers = ['Symbol', 'Type', 'Entry_Price', 'Target_Price', 'Stop_Loss', 'Created_At', 
               'Status', 'Closed_Price', 'Closed_At', 'Profit_Loss_%', 'Duration_Hours', 'Reasons']
    ws1.append([h.replace('_', ' ') for h in headers]) # Use readable headers
    for row in all_signals_data:
        ws1.append([row.get(h) for h in headers])

    # Sheet 2: Active Signals
    ws2 = wb.create_sheet("Active Signals")
    headers_active = ['Symbol', 'Type', 'Entry_Price', 'Current_Price', 'Price_Change_%', 'Created_At', 'Reasons']
    ws2.append([h.replace('_', ' ') for h in headers_active])
    for row in active_signals_data:
        ws2.append([row.get(h) for h in headers_active])

    # Sheet 3: Statistics
    ws3 = wb.create_sheet("Statistics")
    ws3.append(['Metric', 'Value'])
    for stat in stats_data:
        ws3.append([stat['Metric'], stat['Value']])

    # Apply styles to sheets
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")
    center_align = Alignment(horizontal='center', vertical='center')
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), 
                         top=Side(style='thin'), bottom=Side(style='thin'))

    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center_align
            cell.border = thin_border

        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = max(12, min(max_length + 2, 50)) # Min width 12, max 50
            ws.column_dimensions[column].width = adjusted_width
            # Apply border to all cells
            for cell in col:
                cell.border = thin_border

        ws.freeze_panes = ws['A2']

    # Save file
    try:
        wb.save(output_file)
        print(f"Excel report generated: {output_file}")
    except Exception as e:
        print(f"Error saving Excel report: {e}")
        send_telegram_message(f"❌ Error generating Excel report: {e}")
        return

    # Send notification and file to Telegram
    message = (
        f"📊 **Signals Report Generated**\n\n"
        f"🟢 Active Signals: {active_count}\n"
        f"✅ Successful Signals: {target_reached}\n"
        f"❌ Failed Signals: {stop_loss_signals}\n"
        f"📈 **Success Rate:** {success_rate:.2f}%\n"
        f"💰 **Avg P/L:** {avg_profit:.2f}%\n"
        f"📅 Report Time: {datetime.now(tehran_tz).strftime('%Y-%m-%d %H:%M')}"
    )
    if send_telegram_message(message):
        print("Telegram message sent successfully")
    else:
        print("Failed to send Telegram message")

    if send_telegram_file(output_file):
        print("Excel file sent to Telegram successfully")
    else:
        print("Failed to send Excel file to Telegram")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track and report signal status')
    parser.add_argument('--report', action='store_true', help='Generate and send a status report')
    parser.add_argument('--update', action='store_true', help='Only update signal status')
    args = parser.parse_args()

    try:
        if args.report:
            generate_excel_report()
        elif args.update:
            update_signal_status()
        else:
            print("Please specify an action: --report or --update")
    except Exception as e:
        print(f"Error in main execution: {e}")
        send_telegram_message(f"❌ System error in reporting/updating: {e}")
