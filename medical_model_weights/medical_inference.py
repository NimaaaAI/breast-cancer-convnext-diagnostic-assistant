import torch
import logging
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# --- 1. LOGGING CONFIGURATION ---
# Configure logger to output both to terminal and capture model details
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalExpertSystem")

# --- 2. EXPERT MODEL INITIALIZATION ---
# Path to the folder containing config.json and model.safetensors
MODEL_PATH = "./medical_model_weights"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading medical expert model on device: {device}...")
processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# Weight Fingerprint: Digital signature of your specific fine-tuned weights
first_weight_sum = next(model.parameters()).sum().item()
logger.info(f"Model Weight Fingerprint: {first_weight_sum}")
logger.info("Expert model loaded successfully.")

def analyze_ultrasound(image_path):
    """Runs the specialized ConvNeXt model on a medical image."""
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess (matches your Kaggle training phase)
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # --- NEW: RAW MODEL OUTPUT LOGGING ---
        # This converts the tensor to a standard Python list for the logs
        raw_logits_list = logits.cpu().numpy().tolist()
        logger.info(f"RAW MODEL LOGITS (Benign, Malignant): {raw_logits_list}") 
        # -------------------------------------
        
    # Get prediction and confidence
    # Softmax turns raw logits into probabilities that sum to 1.0 (100%)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    conf, classes = torch.max(probs, dim=-1)
    
    # Map labels: Class 0=Benign, Class 1=Malignant
    label_map = {0: "Benign", 1: "Malignant"}
    prediction = label_map[classes.item()]
    confidence_score = conf.item() * 100
    
    return prediction, confidence_score
