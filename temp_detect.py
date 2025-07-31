# temp_detect.py

import serial
import time

# --- IMPORTANT ---
# Make sure this is your Arduino's serial port
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

def get_temperature() -> float | None:

    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            # Give the Arduino time to reset and initialize
            time.sleep(2)
            
            # Read a few lines to find a valid number, discarding intro text
            for _ in range(5): # Try up to 5 times
                line = ser.readline()
                if not line:
                    continue # Skip empty lines

                try:
                    # Decode bytes to string and strip whitespace
                    line_str = line.decode('utf-8').strip()
                    temp = float(line_str)
                    return temp # Success! Return the temperature
                except ValueError:
                    # This line wasn't a valid number, so we'll try the next one
                    print(f"ℹ️ Skipping non-numeric line: '{line_str}'")
                    continue
                    
    except serial.SerialException as e:
        print(f"Error communicating with Arduino: {e}")
        return None
    
    # If we exit the loop without success
    print("Failed to find a valid temperature reading from Arduino.")
    return None

if __name__ == "__main__":
    temp = get_temperature()
    if temp is not None:
        print(f"Current temperature: {temp:.2f}°C")
    else:
        print("Failed to read temperature.")
