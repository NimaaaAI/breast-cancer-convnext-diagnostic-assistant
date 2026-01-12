# 🩺breast-cancer-convnext-diagnostic-assistant
### **Local Medical Inference + Llama 4 Clinical Reporting**

![Python](https://img.shields.io/badge/Python+-blue.svg?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch+-EE4C2C.svg?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange.svg?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama4-red.svg?style=for-the-badge)

---

## 📖 Project Vision
Breast ultrasound imaging is often challenging due to high levels of **"speckle noise"** and artifacts. This project bridges the gap between raw Deep Learning classification and clinical communication. 

By combining a fine-tuned **ConvNeXt-Tiny** "Expert" model with a **Llama 4** "Communicator," we provide a system that not only detects malignancy with high precision but also translates those findings into a structured, professional clinical report.



---

## 🛠️ The Two-Phase Architecture

### **Phase 1: The Expert (Computer Vision)**
* **Model:** `facebook/convnext-tiny-224` (Fine-tuned).
* **Focus:** ROI-optimization to maximize sensitivity to malignant features.
* **Dataset:** Trained on 2,372 clinical ultrasound images.
* **Mapping:** * `Class 0`: **Benign** (Cysts, normal tissue).
    * `Class 1`: **Malignant** (Confirmed cancerous lesions).

### **Phase 2: The Communicator (UI & LLM)**
* **Interface:** **Chainlit** chat interface for seamless image uploads.
* **Brain:** **Llama 4 (Scout-17B)** via the **Groq API** for ultra-fast, structured reporting.
* **Safety:** Automatic handling of low-confidence predictions (<75%) to prevent clinical misinterpretation.

---

## 📸 Demo Preview
When an image is uploaded, the system executes local inference and returns a formatted report:

<img width="1019" height="901" alt="image" src="https://github.com/user-attachments/assets/a09b0787-638a-460a-bf5a-5964b28ce3a6" />


---

## 🚀 Installation & Setup

### 1️⃣ Clone and Prepare Environment
```bash
git clone [https://github.com/NimaaaAI/ROI-Optimized-Breast-Cancer-Classification-with-ConvNeXt.git](https://github.com/NimaaaAI/ROI-Optimized-Breast-Cancer-Classification-with-ConvNeXt.git)
cd ROI-Optimized-Breast-Cancer-Classification-with-ConvNeXt
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt


<img width="968" height="904" alt="image" src="https://github.com/user-attachments/assets/ce84b9cf-273f-4ab9-b0d4-820eb50e96ce" />


python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

running UI:
chainlit run app_ui.py -w
