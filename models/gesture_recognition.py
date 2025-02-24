import tensorflow as tf
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# Gesture to text mapping
gesture_to_text = {
    0: "Hello",
    1: "Thank you",
    2: "Yes",
    3: "No",
    4: "Please",
    5: "Sorry",
    # Add more gestures as needed
}

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(21, 3)),  # 21 landmarks, 3 coordinates each
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(gesture_to_text), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def preprocess_hand_data(hand_landmarks):
    """
    Preprocess hand landmarks data for model input
    Args:
        hand_landmarks: MediaPipe hand landmarks
    Returns:
        numpy array of shape (1, 21, 3) normalized coordinates
    """
    if isinstance(hand_landmarks, landmark_pb2.NormalizedLandmarkList):
        landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] 
                                  for landmark in hand_landmarks.landmark])
    else:
        landmarks_array = np.array(hand_landmarks)
    
    # Normalize to [-1, 1] range if not already normalized
    if np.max(np.abs(landmarks_array)) > 1.0:
        landmarks_array = (landmarks_array - np.mean(landmarks_array)) / np.std(landmarks_array)
    
    # Reshape for model input (batch_size, 21, 3)
    return np.expand_dims(landmarks_array, axis=0)