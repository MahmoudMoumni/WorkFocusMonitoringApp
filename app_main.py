import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np


import os
import cv2
import time
import numpy as np
import mediapipe as mp


import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QSpacerItem, QSizePolicy


# Define the landmark indices for the eyes
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153]
RIGHT_EYE_INDICES = [362, 382, 385, 384, 398, 362, 382, 263, 387, 373, 390]




class VideoCaptureWorker(QThread):
    frame_ready = pyqtSignal(QImage,int,int,int,int)

    def __init__(self, parent=None):
        super(VideoCaptureWorker, self).__init__(parent)
        self._run_flag = False
        # Initialize Mediapipe Face Mesh, Hands, and Drawing modules
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize the Face Mesh model
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize the Hands model
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Define drawing specifications
        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        #load model
        self.saved_model_loaded = tf.saved_model.load("./checkpoints/yolov4-tiny-224", tags=[tag_constants.SERVING])
        self.infer = self.saved_model_loaded.signatures['serving_default']

    def process_frame(self,frame):
        # Padding for the bounding box
        padding = 10  # You can adjust the padding value as needed
        input_size = 224
        is_user_focused=1
        # Flip the frame (e.g., horizontally)
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_bgr_frame=frame.copy()
        image = Image.fromarray(frame)

        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = self.infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=10,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.15
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox
        for i in range(num_boxes[0]):
            class_ind = int(out_classes[0][i])

            if class_ind==67:#cell_phone class index
                #print("cell phone")
                is_user_focused=0

        image = utils.draw_bbox(frame, pred_bbox)
        curr_time = time.time()
        
        # Detect faces in the frame
        face_locations = []
        # Process the frame and detect face mesh  , frame is already rgb
        face_results = self.face_mesh.process(original_bgr_frame)
        # Process the frame and detect hands
        hand_results = self.hands.process(original_bgr_frame)
        
        # Draw the face mesh annotations and bounding box on the frame
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Get the landmark coordinates
                h, w, _ = frame.shape
                landmark_coords = [(int(point.x * w), int(point.y * h)) for point in face_landmarks.landmark]

                # Draw bounding box for left eye
                if all(idx < len(landmark_coords) for idx in LEFT_EYE_INDICES):
                    left_eye_coords = [landmark_coords[idx] for idx in LEFT_EYE_INDICES]
                    x_min = min([coord[0] for coord in left_eye_coords]) - padding
                    y_min = min([coord[1] for coord in left_eye_coords]) - padding
                    x_max = max([coord[0] for coord in left_eye_coords]) + padding
                    y_max = max([coord[1] for coord in left_eye_coords]) + padding
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                else:
                    is_user_focused=0

                # Draw bounding box for right eye
                if all(idx < len(landmark_coords) for idx in RIGHT_EYE_INDICES):
                    right_eye_coords = [landmark_coords[idx] for idx in RIGHT_EYE_INDICES]
                    x_min = min([coord[0] for coord in right_eye_coords]) - padding
                    y_min = min([coord[1] for coord in right_eye_coords]) - padding
                    x_max = max([coord[0] for coord in right_eye_coords]) + padding
                    y_max = max([coord[1] for coord in right_eye_coords]) + padding
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                else:
                    is_user_focused=0

                # Calculate the bounding box coordinates with padding
                x_min = max(min([coord[0] for coord in landmark_coords]) - int(padding/2), 0)
                y_min = max(min([coord[1] for coord in landmark_coords]) - padding, 0)
                x_max = min(max([coord[0] for coord in landmark_coords]) + int(padding/2), w)
                y_max = min(max([coord[1] for coord in landmark_coords]) + padding, h)
                
                # Append coordinates to face_locations in the correct format
                face_locations.append((y_min, x_max, y_max, x_min))
                
                # Draw the bounding box
                #cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        else:
            is_user_focused=0
        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.drawing_spec, self.drawing_spec)
        return frame,int(is_user_focused)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self._run_flag = True
        fps = 0.0
        frame_count = 0
        total_frames=0
        focus_frame_count=0
        start_time = time.time()
        distracted_frames_count=0
        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                frame,is_user_focused=self.process_frame(frame)
                frame_count +=1
                total_frames +=1
                focus_frame_count +=is_user_focused
                focus_percentage=int((focus_frame_count/total_frames)*100)
                if is_user_focused==0:
                    distracted_frames_count +=1
                current_time = time.time()
                # Calculate FPS
                if current_time - start_time >= 1.0:
                    fps = frame_count / (current_time - start_time)
                    start_time = current_time
                    frame_count = 0
                
                # Display the FPS and noise level on the frame
                #cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)     
                image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.frame_ready.emit(image,int(fps),focus_percentage,is_user_focused,distracted_frames_count)
        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class VideoCaptureWidget(QGraphicsView):
    def __init__(self, parent=None):
        super(VideoCaptureWidget, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

    @pyqtSlot(QImage)
    def update_frame(self, image):
        pixmap = QPixmap.fromImage(image)
        self.pixmap_item.setPixmap(pixmap)

class SidebarWidget(QWidget):
    def __init__(self, parent=None):
        super(SidebarWidget, self).__init__(parent)
        layout = QVBoxLayout()

        self.fps_label = QLabel("FPS: 0", self)
        self.focus_time_label = QLabel("Focus Percentage: 0%", self)
        self.phone_time_label = QLabel("Is user working now: ", self)
        self.distracted_time_label = QLabel("Time Distracted: 0s", self)  # New label

        layout.addWidget(self.fps_label)
        layout.addWidget(self.focus_time_label)
        layout.addWidget(self.phone_time_label)
        layout.addWidget(self.distracted_time_label)  # Add new label to layout
        layout.addStretch()
        
        self.setLayout(layout)

    def update_fps(self, fps):
        self.fps_label.setText(f"FPS: {fps}")

    def update_focus_percentage(self, focus_percentage):
        self.focus_time_label.setText(f"Focus Percentage: {focus_percentage}%")

    def update_user_status(self, is_user_focused):
        label_txt = "YES" if is_user_focused else "NO"
        self.phone_time_label.setText(f"Is user working now: {label_txt}")

    def update_distracted_time(self,fps,distracted_frames_count):

        if fps <= 0:
            return
        distracted_total_seconds = distracted_frames_count / fps

        hours = int(distracted_total_seconds // 3600)
        minutes = int((distracted_total_seconds % 3600) // 60)
        seconds = distracted_total_seconds % 60

        self.distracted_time_label.setText(f"Time Distracted: \n {hours} hours, {minutes} minutes, {seconds:.2f} seconds")  # Update new label

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Video Capture App")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QHBoxLayout(self.central_widget)

        self.sidebar = SidebarWidget()
        self.main_layout.addWidget(self.sidebar)

        self.video_layout = QVBoxLayout()
        self.video_widget = VideoCaptureWidget()
        self.video_layout.addWidget(self.video_widget)

        # Create a horizontal layout for the button with spacers
        self.button_layout = QHBoxLayout()
        self.left_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.right_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.button = QPushButton("Start Capture", self)
        self.button.clicked.connect(self.on_button_clicked)

        self.button_layout.addItem(self.left_spacer)
        self.button_layout.addWidget(self.button)
        self.button_layout.addItem(self.right_spacer)

        self.video_layout.addLayout(self.button_layout)

        self.main_layout.addLayout(self.video_layout)

        self.capture_worker = VideoCaptureWorker()
        self.capture_worker.frame_ready.connect(self.update_data)

    def update_data(self, frame, fps, focus_percentage, is_user_focused,distracted_frames_count):  # Add distracted_time parameter
        self.video_widget.update_frame(frame)
        self.sidebar.update_fps(fps)
        self.sidebar.update_focus_percentage(focus_percentage)
        self.sidebar.update_user_status(is_user_focused)
        self.sidebar.update_distracted_time(fps,distracted_frames_count)  # Update the new label

    def on_button_clicked(self):
        if self.button.text() == "Start Capture":
            self.capture_worker.start()
            self.button.setText("Stop Capture")
        else:
            self.capture_worker.stop()
            self.button.setText("Start Capture")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())