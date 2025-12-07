from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

# -------------------- CONFIG --------------------
SOURCE_VIDEO_PATH = "vehicles.mp4"
TARGET_VIDEO_PATH = "result.mp4"
DISPLAY_WIDTH = 1024

# Vidéo originale (modifier selon votre vidéo)
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160

# Ligne de comptage
LINE_Y = int(VIDEO_HEIGHT * 0.69)
LINE_START = sv.Point(0, LINE_Y)
LINE_END = sv.Point(VIDEO_WIDTH, LINE_Y)

DISPLAY_HEIGHT = int((DISPLAY_WIDTH / VIDEO_WIDTH) * VIDEO_HEIGHT)

# -------------------- YOLOv8 --------------------
model = YOLO("yolov8n.pt")
CLASS_NAMES_DICT = model.model.names

SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck']
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[name]
    for name in SELECTED_CLASS_NAMES
]

# -------------------- ByteTrack --------------------
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=25,
    minimum_consecutive_frames=3
)
byte_tracker.reset()

# -------------------- Ligne Zone --------------------
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# -------------------- Annotators --------------------
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=2,
    display_in_count=False,
    display_out_count=False
)

# -------------------- Dashboard Data --------------------
dashboard_data = {
    "in_count": 0,
    "out_count": 0
}


# -------------------- DASHBOARD --------------------
def draw_dashboard(frame, data):

    # Zone rectangle
    overlay = frame.copy()
    alpha = 0.65  # Plus opaque = meilleur contraste

    start_x, start_y = frame.shape[1] - 520, 20
    end_x, end_y = frame.shape[1] - 20, 300

    cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Fonts
    font_title = cv2.FONT_HERSHEY_DUPLEX
    font_text = cv2.FONT_HERSHEY_SIMPLEX


    x = start_x + 20
    y = start_y + 45

    # Titre
    cv2.putText(frame, "Dashboard", (x, y),
            font_title, 1.2, (255, 255, 255), 2)
    y += 45

    # IN
    cv2.putText(frame, f"SUD -> NORD : {data['in_count']}", (x, y),
            font_text, 1.1, (0, 255, 0), 2)
    y += 40

    # OUT
    cv2.putText(frame, f"NORD -> SUD : {data['out_count']}", (x, y),
            font_text, 1.1, (0, 0, 255), 2)
    y += 45

    # Total
    total = data['in_count'] + data['out_count']
    cv2.putText(frame, f"TOTAL : {total}", (x, y),
            font_text, 1.1, (255, 255, 0), 2)

    return frame


# -------------------- CALLBACK --------------------
def callback(frame: np.ndarray, index: int) -> np.ndarray:

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Filtrer classes
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]

    # Tracker
    detections = byte_tracker.update_with_detections(detections)

    # Comptage IN / OUT
    line_zone.trigger(detections)
    dashboard_data["in_count"] = line_zone.in_count
    dashboard_data["out_count"] = line_zone.out_count

    # ---- Annotation image ----
    annotated = frame.copy()
    annotated = trace_annotator.annotate(annotated, detections=detections)
    annotated = box_annotator.annotate(annotated, detections=detections)
    annotated = label_annotator.annotate(annotated, detections=detections)
    annotated = line_zone_annotator.annotate(annotated, line_counter=line_zone)

    # ---- Dashboard ----
    annotated = draw_dashboard(annotated, dashboard_data)

    # Resize display
    display_frame = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    cv2.imshow("YOLOv8 + ByteTrack + Dashboard", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt

    return annotated


# -------------------- PROCESS VIDEO --------------------
try:
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )
finally:
    cv2.destroyAllWindows()
