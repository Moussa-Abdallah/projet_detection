from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# -------------------- CONFIG --------------------
SOURCE_VIDEO_PATH = "vehicles.mp4"
TARGET_VIDEO_PATH = "result.mp4"
DISPLAY_WIDTH = 1024  # largeur pour affichage
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160
LINE_Y = int(VIDEO_HEIGHT * 0.69)  # position verticale de la ligne
LINE_START = sv.Point(0, LINE_Y)
LINE_END = sv.Point(VIDEO_WIDTH, LINE_Y)
DISPLAY_HEIGHT = int((DISPLAY_WIDTH / VIDEO_WIDTH) * VIDEO_HEIGHT)

# Charger le modèle YOLOv8
model = YOLO("yolov8n.pt")
CLASS_NAMES_DICT = model.model.names
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[name]
    for name in SELECTED_CLASS_NAMES
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

# Ligne de comptage
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# Annotateurs
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(thickness=4,text_thickness=4, text_scale=2,display_in_count=False,display_out_count=False)

# -------------------- Dashboard data --------------------
dashboard_data = {
    "in_count": 0,
    "out_count": 0
}

# -------------------- CALLBACK --------------------
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)
    
    # Création des labels
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]
    
    # -------------------- Ligne de comptage --------------------
    crossed_in, crossed_out = line_zone.trigger(detections)
    
    # Mettre à jour le dictionnaire global
    dashboard_data["in_count"] = line_zone.in_count
    dashboard_data["out_count"] = line_zone.out_count
    
    # -------------------- Annotation --------------------
    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)
    annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone, )
    
    # -------------------- Affichage compteur IN/OUT --------------------
    overlay = annotated_frame.copy()
    alpha = 0.7  # fond légèrement plus opaque
    cv2.rectangle(overlay, (20, 20), (360, 140), (30, 30, 30), -1)
    cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)

    font_title = cv2.FONT_HERSHEY_DUPLEX
    font_count = cv2.FONT_HERSHEY_SIMPLEX

    # Titre accrocheur
    cv2.putText(annotated_frame, "Dashboard Comptage Vehicules", (30, 60), font_title, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # IN / OUT plus grands
    cv2.putText(annotated_frame, f"SUD -> NORD : {dashboard_data['in_count']}", (40, 100), font_count, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(annotated_frame, f"NORD -> SUD : {dashboard_data['out_count']}", (40, 135), font_count, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

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
