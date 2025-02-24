from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
from config import Config
from database import init_db, add_feedback_entry
from models.gesture_recognition import create_model, gesture_to_text, preprocess_hand_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
socketio = SocketIO(app)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Initialize model
try:
    model = create_model()
    model.load_weights(Config.MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Initialize database
init_db()

def init_camera():
    """Initialize camera with error handling"""
    try:
        camera = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not camera.isOpened():
            raise Exception("Could not open camera")
        return camera
    except Exception as e:
        logger.error(f"Camera initialization error: {e}")
        return None

def process_frame(frame):
    """Process frame and return processed frame with predictions"""
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        prediction = None
        translated_text = "No gesture detected"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if model is not None:
                    # Preprocess and get prediction
                    processed_data = preprocess_hand_data(hand_landmarks)
                    prediction = model.predict(processed_data)
                    gesture_index = np.argmax(prediction)
                    translated_text = gesture_to_text.get(gesture_index, "Unknown gesture")
                
                # Draw prediction on frame
                cv2.putText(frame, translated_text, (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, prediction, translated_text
    
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return frame, None, "Error processing frame"

def generate_frames():
    """Generate processed frames for video feed"""
    camera = init_camera()
    if camera is None:
        return
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        processed_frame, prediction, translated_text = process_frame(frame)
        
        if prediction is not None:
            socketio.emit('translation_update', {'text': translated_text})
        
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/feedback', methods=['POST'])
def add_feedback():
    try:
        data = request.json
        gesture = data.get('gesture')
        correct_translation = data.get('correct_translation')
        
        if gesture and correct_translation:
            success = add_feedback_entry(gesture, correct_translation)
            if success:
                return jsonify({'status': 'success'})
        
        return jsonify({'status': 'error', 'message': 'Invalid feedback data'})
    
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True)
