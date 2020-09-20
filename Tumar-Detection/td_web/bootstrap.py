from tensorflow.keras.models import load_model

from config import MODEL_DIRECTORY
from tumor_detection.api import TumorDetection

model = load_model(MODEL_DIRECTORY)


def tumor_detection_factory():
    return TumorDetection(model)
