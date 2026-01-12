import os
import base64
import torch
import logging
import chainlit as cl
from groq import Groq
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- 1. LOGGING CONFIGURATION ---
# We configure a logger to provide proof of the expert model's prediction in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalExpertSystem")

# --- 2. EXPERT MODEL INITIALIZATION ---
# Load your fine-tuned ConvNeXt model weights (downloaded from Kaggle)
# Based on your notebook, Class 0 is Benign and Class 1 is Malignant
MODEL_PATH = "./medical_model_weights"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading medical expert model on device: {device}...")
processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
medical_model = AutoModelForImageClassification.from_pretrained(MODEL_PATH).to(device)
medical_model.eval()
logger.info("Expert model loaded successfully.")

first_weight_sum = next(medical_model.parameters()).sum().item()
logger.info(f"Model Weight Fingerprint: {first_weight_sum}")
logger.info("Expert model loaded successfully.")

# --- 3. LLM INITIALIZATION ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_medical_inference(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = medical_model(**inputs)
        logits = outputs.logits
        
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence, class_idx = torch.max(probs, dim=-1)
    
    label_map = {0: "Benign", 1: "Malignant"}
    raw_diagnosis = label_map[class_idx.item()]
    conf_value = confidence.item() * 100

    # NEW: Logic to handle uncertainty
    if conf_value < 75.0:
        diagnosis = f"Inconclusive / Borderline (AI leaning towards {raw_diagnosis})"
    else:
        diagnosis = raw_diagnosis
    
    logger.info(f"PREDICTION: {diagnosis} | CONFIDENCE: {conf_value:.2f}%")
    return diagnosis, conf_value

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="Medical Diagnostic Assistant online. Please upload an ultrasound image.").send()

@cl.on_message
async def main(message: cl.Message):
    # Identify Images in the message
    images = [file for file in message.elements if "image" in file.mime]
    
    llm_prompt = message.content

    if images:
        # STEP A: Local Expert Analysis
        # This creates a visual step in the Chainlit UI while the local model runs
        async with cl.Step(name="Running Expert Local Classification"):
            diagnosis, confidence = run_medical_inference(images[0].path)
        
# STEP B: Construct the prompt for Llama 4
        llm_prompt = f"""
        ### SOURCE DATA BEGIN ###
        Expert Model: ConvNeXt-Tiny (Medical Fine-Tuned)
        Findings: {diagnosis}
        Confidence: {confidence:.2f}%
        Analysis ID: {hash(images[0].path)} 
        ### SOURCE DATA END ###

        USER INQUIRY: {message.content}

        INSTRUCTION: Summarize the findings above for a clinical report. Stay strictly within 
        the bounds of the provided 'Findings' and 'Confidence'. Do not deviate.
        """

    # STEP C: Generate the Final Response via Llama 4 on Groq
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "system", 
                "content": (
                    "STRICT PROTOCOL: You are a medical reporting interface.\n"
                    "1. REPORTING: Format the findings from the [SOURCE DATA] block exactly as provided.\n"
                    "2. UNCERTAINTY: If the 'Findings' contain 'Inconclusive' or 'Borderline', use a "
                    "cautious tone. Emphasize that the AI cannot reach a definitive screening conclusion.\n"
                    "3. NO FABRICATION: Do not change a 'Malignant' finding to 'Benign' or vice-versa. "
                    "Only report what the expert model outputted.\n"
                    "4. MANDATORY DISCLAIMER: Every response must end with: 'NOTE: This is an automated "
                    "preliminary screening. It is NOT a diagnosis. Professional radiological review is required.'"
                )
            },
            {"role": "user", "content": llm_prompt}
        ],
        stream=True,
        temperature=0.1 
    )

    msg = cl.Message(content="")
    for chunk in response:
        if chunk.choices[0].delta.content:
            await msg.stream_token(chunk.choices[0].delta.content)
    
    await msg.send()