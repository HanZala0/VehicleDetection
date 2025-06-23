import streamlit as st
from ultralytics import YOLO
import supervision as sv
import tempfile
import os
import numpy as np
from typing import Optional
from PIL import Image
import cv2

# Set page config
st.set_page_config(page_title="Vehicle Detection System", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("Select Page", ["Vehicle Detection from video", "Bangladeshi Vehicle Detection from image"])

if page == "Vehicle Detection from video":
    st.title("Vehicle Detection, Tracking & Counting")
    st.markdown("Upload a video to detect, track, and count objects using YOLOv8 and Supervision.")

    # Define the ONLY allowed classes
    RESTRICTED_CLASSES = ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]

    # Sidebar options
    with st.sidebar:
        st.header("Configuration")
        model_type = st.selectbox(
            "Select YOLOv8 model",
            ("yolov8n.pt", "yolov8s.pt", "best.pt"),
            index=1  # Default to yolov8s.pt for better speed/accuracy balance
        )
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
        selected_classes = st.multiselect(
            "Select classes to detect",
            options=RESTRICTED_CLASSES,
            default=["car"]
        )
        process_every_n_frames = st.slider("Process every N frames", 1, 10, 2, 
                                          help="Higher values speed up processing but may miss detections")
        resize_factor = st.slider("Resize factor", 0.3, 1.0, 0.5, 0.05,
                                help="Smaller values speed up processing but reduce accuracy")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    def process_video():
        if uploaded_file is None:
            return

        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close()

        # Initialize YOLO model
        model = YOLO(model_type)

        # Get class IDs for selected classes
        if selected_classes:
            class_ids = [
                list(model.model.names.keys())[list(model.model.names.values()).index(cls)] 
                for cls in selected_classes
            ]
        else:
            class_ids = None

        # Initialize ByteTrack tracker
        tracker = sv.ByteTrack()

        # Initialize annotators
        bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        trace_annotator = sv.TraceAnnotator(thickness=2)

        # Open video file
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate resized dimensions
        resized_width = int(width * resize_factor)
        resized_height = int(height * resize_factor)

        # Line counter setup (horizontal line in the middle)
        line_start = sv.Point(0, int(resized_height / 1.33))
        line_end = sv.Point(resized_width, int(resized_height / 1.33))
        line_counter = sv.LineZone(start=line_start, end=line_end)
        line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.8)

        # Streamlit placeholders
        counting_placeholder = st.empty()
        stframe = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0
        processed_frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Skip frames according to processing setting
            if frame_count % process_every_n_frames != 0:
                continue

            processed_frame_count += 1
            status_text.text(f"Processing frame {frame_count}/{total_frames}")

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (resized_width, resized_height))

            # Run YOLO inference
            results = model(small_frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filter detections by confidence and class
            detections = detections[detections.confidence > confidence_threshold]
            if class_ids:
                detections = detections[np.isin(detections.class_id, class_ids)]

            # Update tracker with detections
            detections = tracker.update_with_detections(detections)

            # Check line crossing
            line_counter.trigger(detections)

            # Prepare labels for each detection
            labels = [
                f"ID {tracker_id} {model.model.names[class_id]} {confidence:.2f}"
                for tracker_id, class_id, confidence in
                zip(detections.tracker_id, detections.class_id, detections.confidence)
            ]

            # Annotate frame
            annotated_frame = small_frame.copy()
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            annotated_frame = line_annotator.annotate(annotated_frame, line_counter)

            # Resize back to original dimensions for display
            if resize_factor != 1.0:
                annotated_frame = cv2.resize(annotated_frame, (width, height))

            # Show annotated frame in Streamlit
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        # Release resources
        cap.release()
        
        # Display final counts
        counting_placeholder.markdown(
            f"""
            **Final Count Results:**
            - Total objects entered: {line_counter.in_count}
            - Total objects exited: {line_counter.out_count}
            - Processed {processed_frame_count} of {total_frames} frames
            - Effective processing speed: {original_fps/process_every_n_frames:.1f} FPS
            """
        )

        # Clean up temporary file
        os.unlink(video_path)

    if uploaded_file:
        process_video()

elif page == "Bangladeshi Vehicle Detection from image":
    st.title("Bangladeshi Vehicle Detection")
    st.markdown("Upload an image to detect Bangladeshi vehicles using YOLOv8")
    
    # Sidebar options for Bangladeshi vehicle detection
    with st.sidebar:
        st.header("Configuration")
        confidence_threshold_bd = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01, 
                                          key="bd_confidence")
    
    # Load custom YOLOv8 model for Bangladeshi vehicles
    @st.cache_resource
    def load_bd_model():
        try:
            # Replace with your actual Bangladeshi vehicle detection model
            model = YOLO("best.pt")  # Replace with your custom model path
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    bd_model = load_bd_model()
    
    # Define Bangladeshi vehicle classes (modify according to your model)
    BD_VEHICLE_CLASSES = {
        0: "Ambulance",
        1: "Army Vehicle",
        2: "Auto Rickshaw",
        3: "Bicycle",
        4: "Bus",
        5: "Car",
        6: "Garbagevan",
        7: "Human Hauler",
        8: "Minibus",
        9: "Minivan",
        10: "Motorbike",
        11: "Pickup",
        12: "Policecar",
        13: "Rickshaw",
        14: "Scooter",
        15: "SUV",
        16: "Taxi",
        17: "Three Wheelers -CNG-",
        18: "Truck",
        19: "Van",
        20: "Wheelbarrow"
        #'ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan', 'human hauler', 'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter', 'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
    }
    
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], 
                                    key="bd_image_uploader")
    
    def process_bd_image():
        if uploaded_image is None or bd_model is None:
            return
            
        # Read the image
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if image_np.shape[-1] == 4:  # If image has alpha channel
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        else:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = bd_model(image_np, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Filter detections by confidence
        detections = detections[detections.confidence > confidence_threshold_bd]
        
        # Create labels with custom class names
        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            class_name = BD_VEHICLE_CLASSES.get(class_id, f"Class {class_id}")
            labels.append(f"{class_name} {confidence:.2f}")
        
        # Annotate the image
        box_annotator = sv.BoundingBoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        
        annotated_image = image_np.copy()
        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels)
        
        # Convert back to RGB for display
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(annotated_image, caption="Detected Vehicles", use_column_width=True)
        
        # Display detection summary
        detection_counts = {}
        for class_id in detections.class_id:
            class_name = BD_VEHICLE_CLASSES.get(class_id, f"Class {class_id}")
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
        
        st.subheader("Detection Summary")
        for vehicle_type, count in detection_counts.items():
            st.write(f"- {vehicle_type}: {count}")
    
    if uploaded_image:
        process_bd_image()
