import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
import cv2
import mediapipe as mp
from collections import Counter
import random

class ImprovedWLASLTrainer:
    def __init__(self, videos_dir, nslt_json="nslt_2000.json", classlist_txt="classlist.txt"):
        self.videos_dir = videos_dir
        self.nslt_json = nslt_json
        self.classlist_txt = classlist_txt
        self.label_encoder = LabelEncoder()
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Lower for more detection
            min_tracking_confidence=0.3
        )
        
    def load_class_list(self):
        """Load class index to word mapping"""
        class_to_word = {}
        
        with open(self.classlist_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        class_idx = int(parts[0])
                        word = ' '.join(parts[1:])  # Handle multi-word signs
                        class_to_word[class_idx] = word
        
        print(f"✅ Loaded {len(class_to_word)} class mappings")
        return class_to_word
    
    def load_nslt_data(self):
        """Load NSLT video metadata"""
        with open(self.nslt_json, 'r') as f:
            nslt_data = json.load(f)
        
        print(f"✅ Loaded metadata for {len(nslt_data)} videos")
        return nslt_data
    
    def extract_landmarks_from_video(self, video_path, max_frames=30):
        """Extract hand landmarks from video with better preprocessing"""
        if not os.path.exists(video_path):
            return None
            
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        frame_count = 0
        
        # Get total frames to sample evenly
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > max_frames:
            # Sample frames evenly across the video
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        else:
            frame_indices = list(range(total_frames))
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Preprocess frame
            frame = cv2.resize(frame, (640, 480))  # Consistent size
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Extract landmarks (consistent 126-element format)
            frame_landmarks = [0.0] * 126  # 2 hands × 21 landmarks × 3 coords
            
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_idx >= 2:
                        break
                        
                    hand_coords = []
                    for lm in hand_landmarks.landmark:
                        hand_coords.extend([lm.x, lm.y, lm.z])
                    
                    start_idx = hand_idx * 63
                    frame_landmarks[start_idx:start_idx + 63] = hand_coords
            
            landmarks_sequence.append(frame_landmarks)
        
        cap.release()
        
        # Ensure we have exactly max_frames
        target_length = max_frames
        if len(landmarks_sequence) == 0:
            return None
        elif len(landmarks_sequence) < target_length:
            # Repeat sequence to reach target length
            repeat_times = target_length // len(landmarks_sequence)
            remainder = target_length % len(landmarks_sequence)
            
            extended_sequence = landmarks_sequence * repeat_times
            if remainder > 0:
                extended_sequence.extend(landmarks_sequence[:remainder])
            landmarks_sequence = extended_sequence
        else:
            landmarks_sequence = landmarks_sequence[:target_length]
        
        return np.array(landmarks_sequence)
    
    def balance_dataset(self, X_data, y_labels, max_samples_per_class=15, min_samples_per_class=3):
        """Balance dataset to prevent overfitting to common classes"""
        print(f"🔄 Balancing dataset...")
        
        label_counts = Counter(y_labels)
        print(f"   Original distribution (top 10):")
        for word, count in label_counts.most_common(10):
            print(f"     {word}: {count} samples")
        
        # Group data by label
        label_to_indices = {}
        for i, label in enumerate(y_labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(i)
        
        # Balance each class
        balanced_X = []
        balanced_y = []
        
        kept_classes = 0
        removed_classes = 0
        
        for label, indices in label_to_indices.items():
            if len(indices) < min_samples_per_class:
                print(f"   ❌ Removing '{label}': only {len(indices)} samples")
                removed_classes += 1
                continue
            
            # Limit samples per class
            if len(indices) > max_samples_per_class:
                selected_indices = random.sample(indices, max_samples_per_class)
                print(f"   ⚖️  Limiting '{label}': {len(indices)} → {max_samples_per_class} samples")
            else:
                selected_indices = indices
                print(f"   ✅ Keeping '{label}': {len(indices)} samples")
            
            for idx in selected_indices:
                balanced_X.append(X_data[idx])
                balanced_y.append(y_labels[idx])
            
            kept_classes += 1
        
        print(f"   Final: {kept_classes} classes, {removed_classes} removed")
        print(f"   Total samples: {len(balanced_y)}")
        
        return np.array(balanced_X), np.array(balanced_y)
    
    def augment_data(self, landmarks_sequence):
        """Simple data augmentation"""
        augmented = []
        
        # Original
        augmented.append(landmarks_sequence)
        
        # Add small noise (simulate slight hand position variations)
        noise_factor = 0.02
        noisy = landmarks_sequence + np.random.normal(0, noise_factor, landmarks_sequence.shape)
        augmented.append(noisy)
        
        # Time shift (shift the sequence slightly)
        if len(landmarks_sequence) > 5:
            shift = np.random.randint(1, 3)
            shifted = np.roll(landmarks_sequence, shift, axis=0)
            augmented.append(shifted)
        
        return augmented
    
    def process_dataset(self, limit_videos=None, use_augmentation=True):
        """Process WLASL dataset with improved balancing"""
        print("🚀 Loading class mappings...")
        class_to_word = self.load_class_list()
        
        print("🚀 Loading NSLT metadata...")
        nslt_data = self.load_nslt_data()
        
        X_data = []
        y_labels = []
        processed_count = 0
        failed_count = 0
        
        # Process videos
        video_ids = list(nslt_data.keys())
        if limit_videos:
            video_ids = video_ids[:limit_videos]
            
        print(f"🎬 Processing {len(video_ids)} videos...")
        
        for i, video_id in enumerate(video_ids):
            if i % 100 == 0:
                print(f"   Progress: {i}/{len(video_ids)} videos processed...")
                
            video_info = nslt_data[video_id]
            actions = video_info['action']
            
            # Try different video extensions
            video_path = None
            for ext in ['.mp4', '.avi', '.mov', '.webm']:
                potential_path = os.path.join(self.videos_dir, f"{video_id}{ext}")
                if os.path.exists(potential_path):
                    video_path = potential_path
                    break
            
            if not video_path:
                failed_count += 1
                continue
            
            # Extract landmarks
            landmarks = self.extract_landmarks_from_video(video_path)
            if landmarks is None:
                failed_count += 1
                continue
            
            # For each action in the video
            for action_idx in actions:
                if action_idx in class_to_word:
                    word = class_to_word[action_idx]
                    
                    if use_augmentation and processed_count < 400:  # Augment first 400 for speed
                        # Add augmented versions
                        augmented_sequences = self.augment_data(landmarks)
                        for seq in augmented_sequences:
                            X_data.append(seq)
                            y_labels.append(word)
                            processed_count += 1
                    else:
                        # Just add original
                        X_data.append(landmarks)
                        y_labels.append(word)
                        processed_count += 1
        
        print(f"\n✅ Processing complete:")
        print(f"   Successfully processed: {processed_count} samples")
        print(f"   Failed videos: {failed_count}")
        
        if len(X_data) == 0:
            raise ValueError("❌ No data processed! Check your video paths and files.")
        
        # Balance the dataset
        X_balanced, y_balanced = self.balance_dataset(X_data, y_labels)
        
        return X_balanced, y_balanced, class_to_word
    
    def create_improved_lstm_model(self, input_shape, num_classes):
        """Create improved LSTM model with better regularization"""
        model = Sequential([
            # First LSTM layer
            LSTM(128, return_sequences=True, input_shape=input_shape, 
                 dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Second LSTM layer  
            LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(32, dropout=0.3, recurrent_dropout=0.3),
            BatchNormalization(),
            
            # Dense layers with strong regularization
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'), 
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])
        
        # Use a lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, limit_videos=200, epochs=30):
        """Train the improved model"""
        print("🚀 Starting improved WLASL training...")
        
        # Process dataset
        X, y, class_to_word = self.process_dataset(limit_videos, use_augmentation=True)
        
        print(f"\n📊 Dataset shape: {X.shape}")
        print(f"📊 Unique words: {len(np.unique(y))}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Calculate class weights to handle remaining imbalance
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_encoded), 
            y=y_encoded
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n📈 Training samples: {len(X_train)}")
        print(f"📈 Testing samples: {len(X_test)}")
        
        # Create and train model
        model = self.create_improved_lstm_model(
            input_shape=(X.shape[1], X.shape[2]),
            num_classes=len(np.unique(y_encoded))
        )
        
        print("\n🏗️  Model architecture:")
        model.summary()
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.3, patience=7, min_lr=1e-7, verbose=1)
        ]
        
        # Train with class weights
        print("\n🎯 Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,  # Larger batch size for stability
            callbacks=callbacks,
            class_weight=class_weight_dict,  # Handle class imbalance
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n🎉 Training complete!")
        print(f"🎯 Test Accuracy: {test_accuracy:.4f}")
        
        # Show per-class performance
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
        print(f"\n📈 Class distribution in test set:")
        test_counts = Counter(y_test)
        for class_idx, count in sorted(test_counts.items()):
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            correct = np.sum((y_test == class_idx) & (y_pred == class_idx))
            accuracy = correct / count if count > 0 else 0
            print(f"   {class_name}: {correct}/{count} ({accuracy:.3f})")
        
        return model, history, test_accuracy
    
    def save_model(self, model):
        """Save model and labels"""
        print("💾 Saving model...")
        
        os.makedirs("models", exist_ok=True)
        
        # Save model
        model_path = "models/wlasl_model.h5"
        model.save(model_path)
        
        # Create labels dict
        labels_dict = {}
        for i, label in enumerate(self.label_encoder.classes_):
            labels_dict[str(i)] = label
        
        # Save labels
        labels_path = "models/wlasl_labels.json"
        with open(labels_path, 'w') as f:
            json.dump(labels_dict, f, indent=2)
        
        # Save encoder
        encoder_path = "models/wlasl_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ Saved files:")
        print(f"   {model_path}")
        print(f"   {labels_path}")  
        print(f"   {encoder_path}")
        
        # Show final stats
        print(f"\n📊 Final model stats:")
        print(f"   Total classes: {len(self.label_encoder.classes_)}")
        print(f"   Classes: {', '.join(self.label_encoder.classes_[:10])}{'...' if len(self.label_encoder.classes_) > 10 else ''}")
        
        return model_path, labels_path

def main():
    print("🚀 Improved WLASL Training")
    print("=" * 50)
    
    # Configuration
    videos_dir = input("Enter path to WLASL videos directory: ").strip()
    
    if not os.path.exists(videos_dir):
        print(f"❌ Directory not found: {videos_dir}")
        return
    
    if not os.path.exists("nslt_2000.json"):
        print("❌ nslt_2000.json not found in current directory")
        return
        
    if not os.path.exists("classlist.txt"):
        print("❌ classlist.txt not found in current directory")
        return
    
    # Ask for limits
    try:
        limit = int(input("Limit number of videos for training (default 200): ") or "200")
    except ValueError:
        limit = 1000
    
    print(f"\n🎯 Configuration:")
    print(f"   Videos directory: {videos_dir}")
    print(f"   Video limit: {limit}")
    print(f"   Using: nslt_2000.json")
    
    # Train
    trainer = ImprovedWLASLTrainer(videos_dir)
    
    try:
        model, history, accuracy = trainer.train_model(limit_videos=limit, epochs=30)
        model_path, labels_path = trainer.save_model(model)
        
        print(f"\n🎉 Success! Model trained with {accuracy:.4f} accuracy")
        print("\nTo use with your Flask app:")
        print("   1. Replace your models/ folder with the new one")
        print("   2. Restart your Flask server")
        print("   3. Test with /debug-model endpoint")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()