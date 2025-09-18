from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64
import cv2
import numpy as np
import mediapipe as mp
import os
import json
import tensorflow as tf
from tensorflow import keras
import pickle
from collections import deque
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}}, supports_credentials=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Global variables for WLASL model
wlasl_model = None
wlasl_labels = None
label_encoder = None

# NEW: Sequence buffer for collecting frames over time
sequence_buffer = deque(maxlen=30)  # Store last 30 frames
last_prediction_time = 0
prediction_cooldown = 2.0  # Wait 2 seconds between predictions

def load_wlasl_model():
    """Load WLASL model and labels"""
    global wlasl_model, wlasl_labels, label_encoder
    
    model_path = "models/wlasl_model.h5"
    labels_path = "models/wlasl_labels.json"
    encoder_path = "models/wlasl_encoder.pkl"
    
    try:
        if os.path.exists(model_path) and os.path.exists(labels_path):
            print("🤖 Loading WLASL model...")
            wlasl_model = keras.models.load_model(model_path)
            
            with open(labels_path, 'r') as f:
                wlasl_labels = json.load(f)
            
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
            
            print(f"✅ WLASL model loaded with {len(wlasl_labels)} classes")
            return True
        else:
            print("⚠️  WLASL model files not found. Using basic gesture recognition.")
            return False
    except Exception as e:
        print(f"❌ Error loading WLASL model: {e}")
        return False

def extract_landmarks_sequence(results):
    """Extract landmark sequence for WLASL model prediction"""
    if not results.multi_hand_landmarks:
        return None
    
    frame_landmarks = [0.0] * 126  # 2 hands × 21 landmarks × 3 coords = 126
    
    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        if hand_idx >= 2:  # Only process first 2 hands
            break
            
        hand_coords = []
        for lm in hand_landmarks.landmark:
            hand_coords.extend([lm.x, lm.y, lm.z])
        
        start_idx = hand_idx * 63  # 21 landmarks × 3 coords = 63
        end_idx = start_idx + 63
        frame_landmarks[start_idx:end_idx] = hand_coords
    
    return frame_landmarks

def predict_wlasl_word(sequence):
    """Predict WLASL class using trained model with proper sequence"""
    if not wlasl_model or not wlasl_labels or len(sequence) != 30:
        return None, 0.0
    
    try:
        # Convert sequence to numpy array with proper shape
        features = np.array([sequence])  # (1, 30, 126)
        
        # Make prediction
        predictions = wlasl_model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get the word from labels
        if str(predicted_class) in wlasl_labels:
            word = wlasl_labels[str(predicted_class)]
        else:
            word = "Unknown gesture"
            
        return word, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, 0.0

def reset_sequence_buffer():
    """Reset the sequence buffer"""
    global sequence_buffer
    sequence_buffer.clear()

model_loaded = load_wlasl_model()

@app.route("/")
def home():
    status = "WLASL Model Ready" if model_loaded else "Basic Gestures Only"
    return f"Server running ✅<br>Sign Recognition: {status}"

@app.route("/test")
def test():
    return jsonify({
        "message": "Server working!",
        "success": True,
        "wlasl_model_loaded": model_loaded,
        "model_classes": len(wlasl_labels) if wlasl_labels else 0,
        "sequence_buffer_size": len(sequence_buffer)
    })

@app.route("/sign-to-text", methods=["POST"])
def sign_to_text():
    global last_prediction_time, sequence_buffer
    
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "Invalid request", "success": False}), 400

        # Decode image
        image_data = data['image_data']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Could not decode image", "success": False}), 400

        # Process with MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        current_time = time.time()
        
        if hands_detected > 0:
            # Extract landmarks from current frame
            landmarks = extract_landmarks_sequence(results)
            
            if landmarks:
                # Add to sequence buffer
                sequence_buffer.append(landmarks)
                
                # Only predict if we have enough frames and cooldown period has passed
                if len(sequence_buffer) == 30 and (current_time - last_prediction_time) > prediction_cooldown:
                    if model_loaded:
                        # Convert buffer to list for prediction
                        sequence = list(sequence_buffer)
                        word, confidence = predict_wlasl_word(sequence)
                        
                        if word and confidence > 0.3:  # Increased confidence threshold
                            detected_text = word.title()
                            method = f"WLASL Model (conf: {confidence:.2f})"
                            last_prediction_time = current_time
                            # Clear buffer after successful prediction
                            sequence_buffer.clear()
                        else:
                            detected_text = f"Uncertain gesture (conf: {confidence:.2f})"
                            method = "WLASL Model - Low Confidence"
                    else:
                        detected_text = basic_gesture_recognition(results)
                        method = "Basic Gestures"
                elif len(sequence_buffer) < 30:
                    detected_text = f"Collecting frames... ({len(sequence_buffer)}/30)"
                    method = "Building sequence"
                else:
                    detected_text = f"Waiting for next prediction... ({prediction_cooldown - (current_time - last_prediction_time):.1f}s)"
                    method = "Cooldown period"
            else:
                detected_text = "Could not extract landmarks"
                method = "Error"
        else:
            detected_text = "No hands detected"
            method = "None"
            # Clear buffer when no hands detected
            if len(sequence_buffer) > 0:
                sequence_buffer.clear()

        return jsonify({
            "text": detected_text,
            "success": True,
            "hands_count": hands_detected,
            "method": method,
            "model_loaded": model_loaded,
            "buffer_size": len(sequence_buffer),
            "frames_needed": max(0, 30 - len(sequence_buffer))
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}", "success": False}), 500

@app.route("/reset-buffer", methods=["POST"])
def reset_buffer():
    """Reset the sequence buffer - useful for starting fresh"""
    reset_sequence_buffer()
    return jsonify({
        "success": True,
        "message": "Sequence buffer reset",
        "buffer_size": len(sequence_buffer)
    })

def basic_gesture_recognition(results):
    """Simple fallback gestures"""
    try:
        gestures = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            gestures.append(count_fingers(landmarks))
        return " | ".join(gestures)
    except:
        return "👋 Hand detected"

def count_fingers(landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers_up = 0
    if landmarks[4][0] > landmarks[3][0]:
        fingers_up += 1
    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id][1] < landmarks[tip_id-2][1]:
            fingers_up += 1
    return f"🤚 {fingers_up} fingers detected"

@app.route("/model-info")
def model_info():
    if model_loaded:
        sample_classes = list(wlasl_labels.values())[:10]
        return jsonify({
            "model_loaded": True,
            "total_classes": len(wlasl_labels),
            "sample_classes": sample_classes,
            "model_type": "WLASL LSTM",
            "input_shape": [30, 126],
            "current_buffer_size": len(sequence_buffer),
            "prediction_cooldown": prediction_cooldown
        })
    else:
        return jsonify({
            "model_loaded": False,
            "message": "Using basic gesture recognition",
            "available_gestures": ["finger counting", "basic hand shapes"]
        })

@app.route("/hearing.html")
def serve_hearing():
    return send_from_directory(".", "hearing.html")

@app.route("/index.html")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/style.css")
def serve_css():
    return send_from_directory(".", "style.css")

if __name__ == "__main__":
    print("🚀 Starting WLASL-powered server...")
    print("API endpoints: /, /test, /model-info, /sign-to-text, /reset-buffer")
    if model_loaded:
        print(f"🤖 WLASL model ready with {len(wlasl_labels)} classes")
        print(f"📊 Sequence length: 30 frames, Cooldown: {prediction_cooldown}s")
    else:
        print("⚠️ Using basic gesture recognition (WLASL model not found)")
    app.run(host="0.0.0.0", port=5000, debug=True)