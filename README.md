VWT - Galcanic Security System

A real-time monitoring system using FastAPI, YOLOv8, Tuya smart plugs, and Arduino sensors.
🔧 Features

    Real-time video analysis with YOLOv8 for object detection.

    Web-based FastAPI server with WebSocket support.

    Smart plug automation via Tuya API (e.g., GHome Smart Plug EP2) for controlling lights.

    Temperature monitoring and alert system using data gathered from Arduino Nano 33 BLE Sense Rev2.

    Webcam integration to apply the detection using OpenCV and YOLOv8.

🏗️ Architecture

The architecture of the system involves the following components:

    Arduino Nano 33 BLE Sense Rev2: Used for gathering temperature data. It communicates with the laptop through the serial port using the serialpy library.

    Laptop: Acts as the central server running the FastAPI web application. The laptop also hosts the webcam used for real-time video detection.

    Webcam: Captures live video and applies YOLOv8 object detection via OpenCV.

    Tuya Smart Plug EP2: Controls smart lights that can be toggled remotely via the Tuya API. These can be used to automate actions like turning on/off lights based on events or alerts.

🚀 Quick Start
1. Install dependencies

pip install -r requirements.txt

2. Connect Arduino Nano 33 BLE Sense Rev2

    Connect the Arduino Nano 33 BLE Sense Rev2 to your laptop via the serial port.

    Make sure you have the Arduino IDE and serial drivers installed to facilitate communication.

3. Run the FastAPI app

To run the FastAPI application, use the following command:

uvicorn main:app --reload

Make sure to run this command in the same folder where main.py is located.
4. Smart Plug Setup

To control the GHome Smart Plug EP2 with the Tuya API, ensure the plug is connected and available through the Tuya app. The system will handle the communication and automation of the plug based on certain events (e.g., triggering actions based on temperature or video analysis).
