from tumor_detection.api import TumorDetection
from tensorflow.keras.models import load_model
import os
# Load your trained model
model_directory = os.path.join(os.path.dirname(__file__), "model")
model = load_model(model_directory)




def tumor_detection_factory():
    return TumorDetection(model)
