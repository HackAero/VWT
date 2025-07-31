import os
import json
import time
from datetime import datetime, timedelta
from tuyapy import TuyaApi

import threading

_pulse_thread = None
_pulse_running = False

# Tuya credentials (move to env vars in production)
EMAIL = 'patelvashisth24@gmail.com'
PASSWORD = 'Tensai123'
COUNTRY_CODE = '49'  # Germany
LAST_LOGIN_FILE = 'last_login.json'

api = TuyaApi()
plug = None  # Will hold the plug device


def should_login():
    if not os.path.exists(LAST_LOGIN_FILE):
        return True
    with open(LAST_LOGIN_FILE, 'r') as f:
        data = json.load(f)
        last_login = datetime.fromisoformat(data['last_login'])
        return datetime.now() - last_login > timedelta(seconds=180)


def update_login_timestamp():
    with open(LAST_LOGIN_FILE, 'w') as f:
        json.dump({'last_login': datetime.now().isoformat()}, f)


def init_tuya():
    global plug

    if should_login():
        try:
            print("Logging in to Tuya...")
            api.init(EMAIL, PASSWORD, COUNTRY_CODE)
            update_login_timestamp()
        except Exception as e:
            print(f"Login failed: {e}")
            return False
    else:
        print("Tuya login skipped (within cooldown).")

    try:
        devices = api.get_all_devices()
        if not devices:
            print("No Tuya devices found.")
            return False

        plug = devices[0]
        print(f"Tuya plug loaded: {plug.name()}")
        return True
    except Exception as e:
        print(f"Failed to fetch Tuya devices: {e}")
        return False


def turn_plug_on():
    if plug:
        plug.turn_on()
        return True
    return False


def turn_plug_off():
    if plug:
        plug.turn_off()
        return True
    return False

def pulse_plug(duration_seconds=10):
    global _pulse_thread, _pulse_running

    if _pulse_running:
        print("Plug pulse already running. Skipping.")
        return

    def pulse_loop():
        global _pulse_running
        _pulse_running = True
        print("Starting plug pulse...")
        end_time = time.time() + duration_seconds

        while time.time() < end_time:
            try:
                plug.turn_on()
                time.sleep(1)
                plug.turn_off()
                time.sleep(1)
            except Exception as e:
                print(f"Error during plug pulse: {e}")
                break

        _pulse_running = False
        print("Plug pulse completed.")

    _pulse_thread = threading.Thread(target=pulse_loop, daemon=True)
    _pulse_thread.start()
