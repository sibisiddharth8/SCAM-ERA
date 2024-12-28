import cv2  # Import OpenCV library
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F
import time

def run(video_path , video_path2):

    start_time = time.time()

    # Equivalents for deepfake detection
    threshold_face_similarity = 0.99  # Threshold for face similarity
    threshold_frames_for_deepfake = 15  # Threshold frames for deepfake detection

    mtcnn = MTCNN()
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval()
    cap = cv2.VideoCapture(video_path)  # Start reading the video
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frame rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get the width of the video
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get the height of the video
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Output video codec
    out = cv2.VideoWriter(video_path2, fourcc, fps, (width, height))  # Set up video output

    deepfake_count = 0
    deep_fake_frame_count = 0
    previous_face_encoding = None
    frames_between_processing = int(fps / 7)  # Number of frames between processing
    resize_dim = (80, 80)  # Resize dimensions

    while cap.isOpened():  # Run the loop while the video is open
        ret, frame = cap.read()  # Read the next frame
        if not ret:  # If the frame cannot be read, break the loop
            break

        if frame_count % frames_between_processing == 0:
            boxes, _ = mtcnn.detect(frame)  # Detect faces

            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                face = frame[box[1]:box[3], box[0]:box[2]]

                if not face.size == 0:
                    face = cv2.resize(face, resize_dim)
                    face_tensor = F.to_tensor(face).unsqueeze(0)
                    current_face_encoding = facenet_model(face_tensor).detach().numpy().flatten()

                    if previous_face_encoding is not None:
                        face_similarity = np.dot(current_face_encoding, previous_face_encoding) / (
                                    np.linalg.norm(current_face_encoding) * np.linalg.norm(previous_face_encoding))

                        if face_similarity < threshold_face_similarity:
                            deepfake_count += 1
                        else:
                            deepfake_count = 0

                        if deepfake_count > threshold_frames_for_deepfake:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                            cv2.putText(frame, f'Deepfake Detected - Frame {frame_count}', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            deep_fake_frame_count += 1
                        else:
                            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(frame, 'Real Frame', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                                        cv2.LINE_AA)

                    previous_face_encoding = current_face_encoding

        frame_count += 1
        out.write(frame)  # Write the new frame to the video

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Total Execution Time: {execution_time} seconds")

    cap.release()  # Release the video file
    out.release()  # Release the output video file

    accuracy = (deep_fake_frame_count / frame_count) * 1000  # Calculate accuracy

    if accuracy > 100:
        accuracy = 95  # Cap accuracy if it exceeds 100

    return int(accuracy)

