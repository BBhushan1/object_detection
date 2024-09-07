Pipeline for Person Detection, Classification, and Tracking

1. Imports and Setup
•	Purpose: Initialize the environment by importing the necessary libraries for deep learning, object detection, classification, and tracking.
•	Key Steps:
•	Imported essential libraries such as TensorFlow, PyTorch, OpenCV, NumPy, and more.
•	Configured TensorFlow to dynamically allocate GPU memory to avoid potential memory overloads.
•	Set up PyTorch to utilize either CPU or GPU for optimized performance based on availability.

2. Load Models
•	YOLOv8 Model: Loaded a YOLOv8 model to detect people in each video frame. YOLOv8 is optimized for fast and accurate object detection.
•	Classification Model: Utilized a pre-trained or custom-trained TensorFlow model to classify each detected person as either a "Child" or a "Therapist."
•	ReID Model: Loaded a Re-Identification (ReID) model to generate embeddings for each detected person, helping track individuals across multiple frames by producing unique feature vectors.

3. Initialize Tracker
•	Yolo Inbuilt Tracker: Employed the ByteTrack algorithm for robust tracking. It maintains consistent unique IDs for individuals as they move across the video frames, handling scenarios where individuals leave and re-enter the scene.

4. Preprocessing
•	Purpose: Prepare the input images to be fed into the classification and   ReID models.
•	Key Steps:
•	Resized images to match the input size required by the models.
•	Normalized image pixel values.
•	Adjusted the image format to suit the classification and ReID model requirements .

5. Classification
•	Purpose: Classify each detected person as a "Child" or "Therapist" based on the model's output.
•	Key Steps:
•	Ran the classification model on each detected person’s cropped image.
•	Applied a threshold on the model’s output probabilities to make final predictions, determining whether the person is a "Child" or a "Therapist."

6. Filter Duplicate IDs
•	Purpose: Ensure that each person is assigned a unique ID, even if detected multiple times in different positions.
•	Key Steps:
•	Measured the distance between the centers of detected objects.
•	Removed duplicates by eliminating detections that were too close to one another based on a distance threshold, thereby ensuring unique IDs for each person.

7. Process Video
•	Video Reading: Opened the input video file and prepared a video writer for saving the processed output.
•	Object Detection: Detected people in each frame using the YOLOv8 model, drawing bounding boxes around detected persons.
•	ReID and Tracking: Used the ReID model to compute embeddings for detected persons, and ByteTrack to maintain consistent IDs across frames.
•	Draw Annotations: Annotated each frame with bounding boxes, tracking IDs, and classification labels ("Child" or "Therapist").
•	Save Video: Wrote the processed frames into a new video file, producing a video with labeled bounding boxes, unique tracking IDs, and classification labels.

Summary of Key Components:
1.	Object Detection & Tracking: YOLOv8 detects people, and ByteTrack ensures consistent tracking with unique IDs.
2.	Classification: A TensorFlow-based classification model distinguishes between "Child" and "Therapist" for each detected person.
3.	ReID: A ReID model computes unique embeddings to track individuals across frames, even after occlusions or reappearances.
4.	Duplicate Filtering: Ensures unique ID assignments for people detected multiple times within a close spatial distance.
5.	Video Processing: The video is processed frame-by-frame with detections, classifications, and annotations, and the final output is saved as a new video.


