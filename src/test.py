from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# -------------------- CONFIG --------------------
SOURCE_VIDEO_PATH = "vehicles.mp4"
TARGET_VIDEO_PATH = "result.mp4"
DISPLAY_WIDTH = 1024   # largeur pour l'affichage (plus petit)
DISPLAY_HEIGHT = 576   # hauteur pour l'affichage

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")

# Choix des classes à détecter
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name in SELECTED_CLASS_NAMES
]

# Création du tracker ByteTrack
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=25,  # fps de la vidéo
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# Ligne pour comptage (un peu plus bas)
LINE_START = sv.Point(30, 1100)   # descendue de 100 pixels
LINE_END = sv.Point(3840-30, 1100)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Annotateurs
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# -------------------- CALLBACK --------------------
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)
    
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    
    line_zone.trigger(detections)
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
    
    # Resize pour affichage
    display_frame = cv2.resize(annotated_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("YOLOv8 + ByteTrack", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt
    
    return annotated_frame

# -------------------- PROCESS VIDEO --------------------
try:
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )
finally:
    cv2.destroyAllWindows()