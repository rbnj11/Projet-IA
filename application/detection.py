from ultralytics import YOLO

MODEL_PATH = "yolov8_m.pt"
IMGSZ = 1024  # même valeur que pour l'évaluation

#Seuil détection
CONF = 0.2

# Seuils de la config
DEFAULT_CONF = 0.50
CLASS_CONF = {
    "A5": 0.65, "A15": 0.70,
    "A1": 0.50, "A19": 0.50, "A11": 0.50,
    "A16": 0.25, "A13": 0.25,
}

def load_model(path=MODEL_PATH):
    return YOLO(path)


def predict(model, image):
    """Retourne une liste de détections filtrées par seuil de classe."""
    result = model.predict(image, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    detections = []
    for box in result.boxes:
        name = model.names[int(box.cls)]
        conf = float(box.conf)
        if conf >= CLASS_CONF.get(name, DEFAULT_CONF):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"classe": name, "confiance": round(conf, 3),
                               "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return detections