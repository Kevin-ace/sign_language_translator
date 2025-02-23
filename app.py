from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize camera
def init_camera():
    return cv2.VideoCapture(0)

def process_frame(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    results = hands.process(frame_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    return frame

def generate_frames():
    camera = init_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Convert frame to jpg format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def add_feedback(feedback):
    # Process feedback (e.g., save to database or log)
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.json.get('feedback')
    add_feedback(feedback)
    return jsonify({"status": "success", "message": "Feedback submitted!"})

if __name__ == '__main__':
    app.run(debug=True)
