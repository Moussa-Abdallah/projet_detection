
# Installation – Détection de véhicules (YOLOv8 + ByteTrack + Supervision)

Ce projet utilise **YOLOv8**, **Supervision**, **ByteTrack** et **OpenCV** pour réaliser la détection, le tracking et le comptage de véhicules en temps réel, avec un dashboard intégré.

## 1. Créer et activer un environnement virtuel (recommandé)

```bash
python3 -m venv venv_bt
source venv_bt/bin/activate      # Linux / macOS
venv_bt\Scripts\activate         # Windows
```

---

## 2. Installer les dépendances principales

Ce projet a été testé avec les versions suivantes :

### **Ultralytics – YOLOv8**

```bash
pip install ultralytics==8.3.19
```

### **Supervision (avec les assets et ByteTrack)**

```bash
pip install supervision[assets]==0.24.0
```

---

## 3. Installer les modules indispensables

### OpenCV (traitement d’image)

```bash
pip install opencv-python
```

### Numpy

```bash
pip install numpy
```

---

## 4. Vérifier les installations

```bash
python3 - << 'EOF'
import ultralytics
import supervision
import cv2
import numpy as np

print("Ultralytics version:", ultralytics.__version__)
print("Supervision version:", supervision.__version__)
print("OpenCV version:", cv2.__version__)
EOF
```

---

## 5. Lancer le script principal

Place ta vidéo dans le dossier du projet, puis exécute :

```bash
python3 detect.py
```
