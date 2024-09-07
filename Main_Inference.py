import cv2
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict
from ultralytics import YOLO
import torch
import torchreid
from sklearn.metrics.pairwise import cosine_similarity


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU detected and dynamic memory allocation enabled.")
    except Exception as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU detected, running on CPU.")


try:
    yolo_model = YOLO('yolov8x.pt')  
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Tried to implemented a image classifier in between inference although the model is still need to be fine-tunned
model_path = r'D:\CogniAble\Object_Tracking\models\ModelE.h5' 
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit()
classification_model = tf.keras.models.load_model(model_path)


classification_model.summary()


def load_reid_model(model_name='osnet_x0_25'):
    """Load the ReID model."""
    reid_model = torchreid.models.build_model(
        name=model_name,
        num_classes=1,  
        pretrained=True
    )
    reid_model.eval()  
    return reid_model

reid_model = load_reid_model()
print("ReID model loaded successfully.")

def preprocess_for_classification(cropped_image):
    """Preprocess the image before classification."""
    resized_image = cv2.resize(cropped_image, (256, 256))  
    normalized_image = resized_image / 255.0  
    return np.expand_dims(normalized_image, axis=0)  

def classify_person(cropped_image):
    """Classify if the person is a child or an adult."""
    preprocessed_image = preprocess_for_classification(cropped_image)
    predictions = classification_model.predict(preprocessed_image)
    
    
    print(f"Raw predictions: {predictions}")
    
    if predictions.shape[1] == 1:  
        probability = predictions[0][0]  
        print(f"Predicted probability: {probability}")
        
        threshold = 0.6
        return "Child" if probability > threshold else "Therapist"
    else:
        raise ValueError("Unexpected prediction shape.")

def filter_duplicate_ids(boxes, track_ids, distance_threshold=50):
    """Filter out duplicate IDs based on distance between object centers."""
    filtered_ids = []
    filtered_boxes = []

    for i, (box1, id1) in enumerate(zip(boxes, track_ids)):
        duplicate = False
        x1, y1, w, h = box1
        center1 = (x1 + w / 2, y1 + h / 2)
        for j, (box2, id2) in enumerate(zip(filtered_boxes, filtered_ids)):
            x2, y2, w2, h2 = box2
            center2 = (x2 + w2 / 2, y2 + h2 / 2)
            distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
            if distance < distance_threshold:
                duplicate = True
                break
        if not duplicate:
            filtered_ids.append(id1)
            filtered_boxes.append(box1)
    
    return filtered_boxes, filtered_ids

def extract_features(image, model):
    """Extract features from the image using the ReID model."""
    image = cv2.resize(image, (256, 256))  
    image = np.transpose(image, (2, 0, 1))  
    image = torch.tensor(image).float().unsqueeze(0)  

    with torch.no_grad():
        features = model(image)
    return features


def match_track_by_features(new_features, track_features, threshold=0.5):
    """Match a new set of features with the existing track features based on cosine similarity."""
    for track_id, features in track_features.items():
        similarity = cosine_similarity(new_features.numpy(), features.numpy())
        if similarity >= threshold:  
            return track_id  
    return None

def process_video(input_video_path, output_video_path):
    """Process a single video file and save the output."""
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    track_history = defaultdict(lambda: [])
    biometric_labels = {}
    track_features = {}  
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error reading frame.")
            break

        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}")
        
        results = yolo_model.track(frame, classes=[0], persist=True, save=False, tracker="bytetrack.yaml", conf=0.3, iou=0.45, nms=True)

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id
            if track_ids is not None:
                track_ids = track_ids.int().cpu().tolist()
                filtered_boxes, filtered_ids = filter_duplicate_ids(boxes, track_ids)

                for box, track_id in zip(filtered_boxes, filtered_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)

                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    cropped_image = frame[y1:y2, x1:x2]

                    if track_id not in biometric_labels:
                        try:
                            label = classify_person(cropped_image)
                            biometric_labels[track_id] = label  

                            new_features = extract_features(cropped_image, reid_model)
                            matched_track_id = match_track_by_features(new_features, track_features)

                            if matched_track_id is not None:
                                track_id = matched_track_id
                                biometric_labels[track_id] = label  
                            else:
                                track_features[track_id] = new_features  
                        except ValueError as e:
                            print(f"Error in classification: {e}")

                for i, track_id in enumerate(filtered_ids):
                    if track_id in biometric_labels:
                        label = f"{biometric_labels[track_id]} ID: {track_id}"
                        x, y, w, h = filtered_boxes[i]
                        color = (0, 255, 0) if biometric_labels[track_id] == "Child" else (255, 0, 0)
                        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
                        cv2.putText(frame, label, (int(x - w / 2), int(y - h / 2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                out.write(frame)

    cap.release()
    out.release()

    print(f"Video saved successfully: {output_video_path}")


input_video_path = r'D:\CogniAble\Object_Tracking\Input\Natural Environment Teaching (NET).mp4'
output_video_path = r'D:\CogniAble\Object_Tracking\output\Natural Environment Teaching (NET).mp4'


process_video(input_video_path, output_video_path)
