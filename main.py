
import asyncio

import json

import random

import cv2

import numpy as np

from ultralytics import YOLO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse

from pathlib import Path

from contextlib import asynccontextmanager

from collections import Counter

from temp_detect import get_temperature


from tuya_control import init_tuya, turn_plug_on, turn_plug_off, pulse_plug

import pygame

import time

import threading

# Initialize pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")  

# Cooldown to prevent continuous beeping
last_alarm_time = 0
alarm_cooldown_seconds = 5
recent_alert_boxes = []
recent_box_memory_seconds = 3  


# -------- CONFIGURATION --------

CONFIDENCE_THRESHOLD = 0.3

MODEL_PATH = "yolov8n.pt" 

VIDEO_SOURCE = 0 


model = None

camera = None


# -------- LIFESPAN MANAGEMENT --------

@asynccontextmanager

async def lifespan(app: FastAPI):
    global model, camera

    print("INFO:     Application startup...")

    try:
        model = YOLO(MODEL_PATH)
        print(f"YOLO Detection model loaded successfully from '{MODEL_PATH}'.")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")

    camera = cv2.VideoCapture(VIDEO_SOURCE)
    if camera.isOpened():
        print(f"Camera at source '{VIDEO_SOURCE}' opened successfully.")
    else:
        print(f"Failed to open camera at source '{VIDEO_SOURCE}'.")

    # Initialize Tuya Smart Plug
    init_tuya()

    asyncio.create_task(stream_realistic_temperature_data())

    yield

    print("INFO:     Application shutdown...")

    if camera and camera.isOpened():
        camera.release()
        print("Camera released.")


# -------- HELPER FUNCTION FOR BOUNDING BOX DISTANCE

def boxes_are_close(boxA, boxB, threshold=100):
    """
    Returns True if the distance between the edges of two boxes is less than the threshold (in pixels).
    """
    # Unpack boxes
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    # Horizontal distance
    dx = max(0, max(bx1 - ax2, ax1 - bx2))
    # Vertical distance
    dy = max(0, max(by1 - ay2, ay1 - by2))

    # Distance between edges (not centroids)
    distance = np.hypot(dx, dy)
    return distance < threshold

# -------- INITIALIZATION --------

app = FastAPI(

    title="Safety Dashboard Backend",

    description="Provides real-time data streams for PPE detection and environmental monitoring.",

    version="2.8.0", # Updated version for black color focus

    lifespan=lifespan

)


# --- WebSocket Connection Managers ---

class ConnectionManager:

    def __init__(self):

        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):

        await websocket.accept()

        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):

        if websocket in self.active_connections:

            self.active_connections.remove(websocket)

    async def broadcast_json(self, data: dict):

        message = json.dumps(data)

        for connection in self.active_connections:

            await connection.send_text(message)

    async def broadcast_text(self, text: str):

        for connection in self.active_connections:

            await connection.send_text(text)


temp_manager = ConnectionManager()

ppe_manager = ConnectionManager()


# -------- COLOR DETECTION HELPER FUNCTIONS --------


def get_dominant_color(image, k=4):

    """

    Finds the dominant color in an image using k-means clustering.

    Returns the dominant color as a BGR tuple.

    """

    if image is None or image.size == 0:

        return None


    pixels = image.reshape((-1, 3))

    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    most_common_label = Counter(labels.flatten()).most_common(1)[0][0]

    dominant_color = np.uint8(centers[most_common_label])

    return tuple(dominant_color.tolist())


def is_black_shirt(bgr_color, luminance_threshold=60):
    """
    Determines if a BGR color is 'black' based on its luminance (brightness).
    A lower luminance means a darker color.
    """
    if bgr_color is None:
        return False

    # Convert BGR to grayscale luminance using weighted sum
    b, g, r = bgr_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    return luminance < luminance_threshold



# -------- CORE DETECTION LOGIC WITH BLACK COLOR FOCUS --------


def analyze_frame(model, frame, confidence_threshold=0.3):
    global recent_alert_boxes
    now = time.time()
    recent_alert_boxes = [(t, c) for t, c in recent_alert_boxes if now - t < recent_box_memory_seconds]
    
    results = model(frame, verbose=False)[0]
    persons = []
    ppe_stats = {"black": 0, "not_black": 0, "total": 0}

    # --- QR Code Detection using cv2.QRCodeDetector ---
    qr_box = None
    qr_detector = cv2.QRCodeDetector()
    qr_data, points, _ = qr_detector.detectAndDecode(frame)
    qr_box = None

    if qr_data and points is not None:
        points = points[0]  # shape (4, 2)
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        x1, y1 = int(x_coords.min()), int(y_coords.min())
        x2, y2 = int(x_coords.max()), int(y_coords.max())
        qr_box = (x1, y1, x2, y2)

    # Draw QR bounding box
        cv2.polylines(frame, [np.int32(points)], isClosed=True, color=(255, 0, 255), thickness=2)
        cv2.putText(frame, "QR Code", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


    # --- Person Detection ---
    for box in results.boxes:
        try:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if model.names[cls] == "person" and conf > confidence_threshold:
                persons.append(box)
        except (IndexError, ValueError):
            continue

    annotated_frame = frame.copy()
    ppe_stats["total"] = len(persons)

    for person_box in persons:
        x1, y1, x2, y2 = map(int, person_box.xyxy[0])
        box_width, box_height = x2 - x1, y2 - y1

        # Torso ROI
        roi_x_start = x1 + int(box_width * 0.20)
        roi_x_end = x2 - int(box_width * 0.20)
        roi_y_start = y1 + int(box_height * 0.20)
        roi_y_end = y1 + int(box_height * 0.60)
        torso_roi = annotated_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        if torso_roi.size == 0:
            continue

        dominant_bgr = get_dominant_color(torso_roi)
        person_bbox = (x1, y1, x2, y2)

        # Check proximity to QR code
        if qr_box and boxes_are_close(person_bbox, qr_box, threshold=30):
            label = "Too close to danger liquids"
            color = (0, 0, 255)  # Red

            # Compute person box center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            person_center = (center_x, center_y)

            # Check if this person has already triggered the alarm recently
            should_alert = True
            now = time.time()
            for previous_time, previous_center in recent_alert_boxes:
                distance = np.hypot(center_x - previous_center[0], center_y - previous_center[1])
                if distance < 50 and now - previous_time < recent_box_memory_seconds:
                    should_alert = False
                    break

            # If it's a new person or enough time has passed, trigger alar
            if should_alert:
                recent_alert_boxes.append((now, person_center))
                global last_alarm_time
                if now - last_alarm_time >= alarm_cooldown_seconds:
                    last_alarm_time = now
                    threading.Thread(target=alarm_sound.play, daemon=True).start()
                    threading.Thread(target=pulse_plug, kwargs={"duration_seconds":10}, daemon=True).start()



            # Draw translucent red box
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            annotated_frame = cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0)


        else:
            if is_black_shirt(dominant_bgr):
                label = "Protective gear"
                color = (0, 255, 0)  # Green
                ppe_stats["black"] += 1
            else:
                label = "No Protective gear"
                color = (0, 165, 255)  # Orange
                ppe_stats["not_black"] += 1

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0] + 10, y1), color, cv2.FILLED)
        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated_frame, ppe_stats




# -------- BACKGROUND STREAMING TASKS --------

async def stream_video_and_ppe_data():

    global model, camera

    print("Video and PPE data streamer has started.")

    while True:

        if camera is None or not camera.isOpened() or model is None:

            placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)

            text = "Camera not available" if camera is None or not camera.isOpened() else "Model not loaded"

            cv2.putText(placeholder_frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', placeholder_frame)

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'

                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            await asyncio.sleep(1)

            continue


        success, frame = camera.read()

        if not success:

            await asyncio.sleep(0.1)

            continue

        

        annotated_frame, ppe_stats = analyze_frame(model, frame, CONFIDENCE_THRESHOLD)

        await ppe_manager.broadcast_json(ppe_stats)

        

        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        if not ret:

            continue

        

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        

        await asyncio.sleep(0.03)


async def stream_realistic_temperature_data():

    print("Real temperature streamer has started.")
    
    while True:
        current_temp = await asyncio.to_thread(get_temperature)

        if current_temp is not None:
            await temp_manager.broadcast_text(f"{current_temp:.2f}")

        await asyncio.sleep(2.0)
        
        
async def stream_depth_feed():
    global pipeline
    print("Depth feed streamer started.")
    color_map = cv2.COLORMAP_JET

    while True:
        if pipeline is None:
            await asyncio.sleep(1)
            continue

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            await asyncio.sleep(0.03)
            continue

        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colored = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), color_map)

        ret, buffer = cv2.imencode('.jpg', depth_colored)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        await asyncio.sleep(0.03)
        
 



# -------- API ROUTES --------

@app.get("/", response_class=HTMLResponse)

async def get_dashboard():

    """Serves the main HTML dashboard page."""

    html_file_path = Path(__file__).parent / "index.html"

    if html_file_path.is_file():

        return FileResponse(html_file_path)

    return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=404)


@app.get("/video_feed")

async def video_feed():

    """Provides the annotated video stream."""

    return StreamingResponse(

        stream_video_and_ppe_data(),

        media_type="multipart/x-mixed-replace; boundary=frame"

    )


@app.websocket("/ws/temperature")

async def websocket_temperature_endpoint(websocket: WebSocket):

    """Handles WebSocket connections for temperature data."""

    await temp_manager.connect(websocket)

    print(f"Client connected to temperature stream. Total clients: {len(temp_manager.active_connections)}")

    try:

        while True:

            await websocket.receive_text()

    except WebSocketDisconnect:

        temp_manager.disconnect(websocket)

        print(f"Client disconnected from temperature stream. Total clients: {len(temp_manager.active_connections)}")


@app.websocket("/ws/ppe_stats")

async def websocket_ppe_endpoint(websocket: WebSocket):

    """Handles WebSocket connections for PPE detection statistics."""

    await ppe_manager.connect(websocket)

    print(f"Client connected to PPE stats stream. Total clients: {len(ppe_manager.active_connections)}")

    try:

        while True:

            await websocket.receive_text()

    except WebSocketDisconnect:

        ppe_manager.disconnect(websocket)

        print(f"Client disconnected from PPE stats stream. Total clients: {len(ppe_manager.active_connections)}")
        

@app.post("/plug/on")
async def plug_on():
    success = await asyncio.to_thread(turn_plug_on)
    return {"status": "on" if success else "failed"}

@app.post("/plug/off")
async def plug_off():
    success = await asyncio.to_thread(turn_plug_off)
    return {"status": "off" if success else "failed"}


