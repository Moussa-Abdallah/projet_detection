from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# -------------------- CONFIG --------------------
SOURCE_VIDEO_PATH = "vehicles.mp4"
TARGET_VIDEO_PATH = "result.mp4"

# Largeur pour affichage, hauteur calculée automatiquement
DISPLAY_WIDTH = 1024

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
    frame_rate=25,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# Ligne pour comptage (plus haut, sur toute la largeur)
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160

# Distance depuis le bas actuelle : 10% de la hauteur
# Double cette distance : 20% depuis le bas
LINE_Y = int(VIDEO_HEIGHT * 0.69)  # 80% de la hauteur originale
LINE_START = sv.Point(0, LINE_Y)          # toute la largeur
LINE_END = sv.Point(VIDEO_WIDTH, LINE_Y)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Calcul automatique de la hauteur pour garder le ratio
DISPLAY_HEIGHT = int((DISPLAY_WIDTH / VIDEO_WIDTH) * VIDEO_HEIGHT)

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
    
    
    # Resize pour affichage en conservant la largeur
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
