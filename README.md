# 🧠 MediCare-AI  
**Multimodal Alzheimer’s Disease Detection and Patient Assistance System**

MediCare-AI is a Flask-based web application developed as a Final Year Project (FYP) for multimodal Alzheimer’s disease detection and patient support. It combines medical image preprocessing, deep learning-based inference, and an AI-powered chatbot to provide a structured and research-oriented solution for Alzheimer’s analysis.

The system is designed with real-world challenges in mind, including limited medical data, computational constraints, and the importance of ethical AI in healthcare. It focuses on building a practical, modular, and deployable medical AI pipeline rather than just a theoretical model.

---

## 📌 Project Overview

Alzheimer’s disease is a progressive neurological disorder where early detection can play a critical role. MediCare-AI aims to support early-stage analysis by processing MRI and PET scans using a deep learning-based multimodal approach.

The application provides:
- A clean and user-friendly web interface  
- Secure patient interaction and data handling  
- A structured preprocessing and inference pipeline  
- AI-based assistance through a chatbot (MediBot)  
- A modular architecture for future research and improvements  

---

## 🎯 Project Objectives

- Analyze MRI and PET scans for Alzheimer’s-related patterns  
- Build a robust preprocessing and inference pipeline  
- Develop a multimodal deep learning framework  
- Provide AI-based assistance through a medical chatbot  
- Maintain structured records of uploaded scans  
- Design a scalable system for future research extensions  

---

## ✨ Core Features

### 🔐 User Authentication
- Secure patient login system  
- Session-based authentication using Flask  
- Protected routes for user-specific operations  

---

### 🧠 Medical Image Analysis
- MRI and PET scan upload support  
- Image preprocessing pipeline (normalization, preparation)  
- Deep learning-based inference using trained models  
- Efficient model loading at application startup  

---

### 🤖 MediBot (Medical Chat Assistant)
- Alzheimer’s awareness and educational guidance  
- Prevention tips and lifestyle recommendations  
- Context-based responses using a knowledge-driven approach  
- Designed to assist, not diagnose  

---

### 📂 Patient Data Handling
- Secure handling of uploaded medical images  
- Organized file storage system  
- Separation of user data and application logic  

---

### 🏗 Modular Architecture
- Clear separation of components:
  - Data ingestion  
  - Preprocessing  
  - Model inference  
  - User interface  
- Easy to maintain, debug, and extend  
- Suitable for experimentation and research work  

---

## 🧩 System Workflow

1. User logs into the system  
2. MRI or PET scan is uploaded  
3. Image preprocessing is applied  
4. The trained deep learning model performs inference  
5. Results are displayed to the user  
6. MediBot provides awareness and guidance  

---

## 🛠 Tech Stack

### Backend
- Flask  
- Python 3.10+  
- Jinja2 Templates  
- WTForms / Flask form handling  
- Session-based authentication  

---

### AI / ML
- CNN-based deep learning model  
- Multimodal design (MRI + PET ready)  
- Medical image preprocessing pipeline  
- Model inference optimization (load once, reuse)  

---

### Storage
- Local filesystem (uploads, processed files)  
- SQLite (lightweight database)  
- External model storage (for large `.pth` files)  

---

### Deployment
- GitHub (source code management)  
- Hugging Face Hub (model storage)  
- Render (application deployment)  

---

## 🧠 Model Information

The system uses a trained deep learning fusion model:

- `final_fusion_model.pth`

Large model files are not stored in the GitHub repository due to size limits.  
Instead, they are hosted externally and loaded dynamically when the application starts.

---

## 📁 Project Structure

```bash
fyp_implementation/
│
├── main.py
├── uploads_model.py
├── requirements.txt / pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
│
├── templates/
├── static/
│
├── models/          # not included in GitHub
├── data/            # not included in GitHub
├── instance/
└── __pycache__/
```
## ⚙️ Setup & Installation
1) Clone the Repository
``` bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
2) Create Virtual Environment
```bash
python -m venv .venv
```
4) Activate Virtual Environment
Windows
```bash
.venv\Scripts\activate
```
Linux / macOS
```bash
source .venv/bin/activate
```
6) Install Dependencies

Using uv:
```bash
uv sync
```
Using pip:
```python
pip install -r requirements.txt
```
## 🔑 Environment Variables

Create a .env file in the root directory:
```bash
SECRET_KEY=your_secret_key
HUGGINGFACE_TOKEN=your_token_if_needed
DATABASE_URL=your_database_url
```
Important:

Do NOT upload .env to GitHub
Add .env to .gitignore
🚀 Running the Application
python main.py

Open your browser and access the local server.

## ☁️ Deployment Notes
Keep source code on GitHub
Store large models on external platforms (e.g., Hugging Face Hub)
Load models at application startup
Use environment variables for sensitive data
Deploy using platforms like Render

## 🔄 Model Loading Strategy
Check if the model exists locally
If not, download from external storage
Load the model once at startup
Reuse the loaded model for all predictions

This improves performance and avoids repeated loading overhead.

## 📊 Project Significance

This project demonstrates:

Practical medical AI system design
Multimodal deep learning approach
Real-world deployment considerations
Clean and modular software architecture
Integration of AI with web applications

It goes beyond a simple model by building a complete, deployable AI system.

## ⚠ Disclaimer

MediCare-AI is intended for educational and research purposes only.

It does not provide medical diagnosis and must not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.

## 👨‍💻 Author

Muhammad Zahraan
AI Graduate | Engineer | 
Final Year Project on Multimodal Alzheimer’s Disease Detection using Deep Learning
