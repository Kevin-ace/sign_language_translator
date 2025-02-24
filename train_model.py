import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from models.gesture_recognition import create_model, preprocess_hand_data, gesture_to_text
import os
import time
from tqdm import tqdm

def draw_instruction_box(frame, text, box_color=(50, 50, 50), text_color=(255, 255, 255)):
    """Draw a semi-transparent instruction box with text"""
    h, w = frame.shape[:2]
    # Create overlay for semi-transparent box
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h-120), (w-10, h-20), box_color, -1)
    # Add transparency
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Split text into lines for better display
    lines = text.split('\n')
    y_position = h-100
    for line in lines:
        cv2.putText(frame, line, (20, y_position), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        y_position += 30
    
    return frame

def collect_training_data():
    """Collect training data using webcam with enhanced visual feedback"""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    X = []
    y = []
    
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None, None

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    total_gestures = len(gesture_to_text)
    
    for gesture_id, gesture_name in gesture_to_text.items():
        print(f"\nPreparing to collect data for gesture {gesture_id + 1}/{total_gestures}: {gesture_name}")
        
        # Preparation phase
        countdown_time = 5
        while countdown_time >= 0:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process hand landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Show preparation instructions
            instructions = f"Prepare to show gesture: {gesture_name}\n"
            instructions += f"Starting in: {countdown_time} seconds\n"
            instructions += "Position your hand in frame and press 'c' to start"
            frame = draw_instruction_box(frame, instructions)
            
            cv2.imshow('Collecting Data', frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('c'):
                break
            elif key & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, None
            
            time.sleep(1)
            countdown_time -= 1
        
        # Collection phase
        frames_collected = 0
        collection_start_time = time.time()
        
        # Progress bar for collection
        pbar = tqdm(total=100, desc=f"Collecting {gesture_name}")
        
        while frames_collected < 100:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Process data
                processed_data = preprocess_hand_data(hand_landmarks)
                X.append(processed_data[0])
                y.append(gesture_id)
                frames_collected += 1
                pbar.update(1)
                
                # Calculate collection speed
                elapsed_time = time.time() - collection_start_time
                fps = frames_collected / elapsed_time if elapsed_time > 0 else 0
                
                # Show collection progress
                status = f"Collecting: {gesture_name}\n"
                status += f"Progress: {frames_collected}/100 frames\n"
                status += f"Collection speed: {fps:.1f} fps"
                frame = draw_instruction_box(frame, status, 
                                          box_color=(0, 100, 0), 
                                          text_color=(255, 255, 255))
            else:
                # Warning if hand not detected
                frame = draw_instruction_box(frame, 
                                          "⚠️ No hand detected!\nPlease keep your hand in frame",
                                          box_color=(0, 0, 100),
                                          text_color=(255, 255, 255))
            
            # Show progress bar on frame
            progress_width = int((frames_collected / 100) * frame.shape[1])
            cv2.rectangle(frame, (0, 0), (progress_width, 10), (0, 255, 0), -1)
            
            cv2.imshow('Collecting Data', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, None
        
        pbar.close()
        
        # Short break between gestures
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if ret:
                frame = draw_instruction_box(frame, 
                                          f"Great! Taking a break...\nNext gesture in {i} seconds",
                                          box_color=(0, 100, 100))
                cv2.imshow('Collecting Data', frame)
                cv2.waitKey(1)
            time.sleep(1)
    
    cap.release()
    cv2.destroyAllWindows()
    
    return np.array(X), np.array(y)

def train_model():
    """Train the model with progress visualization"""
    print("\n=== Starting Data Collection ===")
    print("Please follow the on-screen instructions.")
    print("Press 'q' at any time to quit.")
    
    # Collect data
    X, y = collect_training_data()
    
    if X is None or y is None:
        print("\nData collection cancelled.")
        return None, None
    
    print("\n=== Data Collection Complete ===")
    print(f"Collected {len(X)} samples across {len(gesture_to_text)} gestures")
    
    # Convert labels to one-hot encoding
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=len(gesture_to_text))
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    print("\n=== Training Model ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create and train model
    model = create_model()
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/weights/gesture_model.h5',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.2,
            patience=3,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n=== Final Evaluation ===")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    return model, history

if __name__ == "__main__":
    # Create weights directory if it doesn't exist
    os.makedirs('models/weights', exist_ok=True)
    
    print("=== Sign Language Gesture Recognition Training ===")
    print("This script will help you collect training data and train the model.")
    print(f"Gestures to collect: {list(gesture_to_text.values())}")
    input("\nPress Enter to start...")
    
    # Train the model
    model, history = train_model()
    
    if model is not None:
        print("\n=== Training Complete ===")
        print("Model saved to: models/weights/gesture_model.h5")
        print("\nYou can now run the main application with 'python app.py'")
