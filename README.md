# 🧠 MediCare-AI  
**Multimodal Alzheimer’s Disease Detection and Assistance System**

MediCare-AI is a Flask-based web application designed for Alzheimer’s Disease analysis and patient assistance.  
The system integrates medical image preprocessing, deep learning models, and an AI-powered medical chatbot to support awareness, early-stage analysis, and patient guidance.

This project is developed as a Final Year Project (FYP) with emphasis on practical medical AI challenges, limited data availability, computational constraints, and ethical system design.

---

## 🎯 Project Objectives

- Analyze MRI and PET scans for Alzheimer’s Disease-related patterns  
- Build a structured preprocessing and inference pipeline  
- Assist patients using an AI-based medical chatbot  
- Maintain uploaded scan records for future analysis  
- Design a modular system suitable for research extension  

---

## ✨ Core Features

- 🔐 **User Authentication**  
  - Patient login system  
  - Secure session handling using Flask  

- 🧠 **Medical Image Analysis**  
  - MRI and PET scan upload support  
  - Image preprocessing pipeline  
  - CNN-based deep learning inference  

- 🤖 **MediBot (Medical Chatbot)**  
  - Alzheimer’s awareness and guidance  
  - Prevention tips and lifestyle suggestions  
  - Knowledge-based, non-diagnostic responses  

- 📂 **Patient Data Handling**  
  - Secure file uploads  
  - Organized storage for scans and embeddings  

- 🏗 **Modular Architecture**  
  - Clear separation of ingestion, preprocessing, models, and UI  
  - Easy modification for experimentation and research  

---

## 🛠 Tech Stack

### Backend
- Flask  
- Python 3.10+  
- Jinja2 Templates  
- WTForms  
- Session-based authentication  

### AI / ML
- CNN-based deep learning models  
- Medical image preprocessing  
- Multimodal-ready design (MRI + PET)  
- Vector store for chatbot context  

### Storage
- Local filesystem (uploads, vectorstore)  
- SQLite (local database)  

---

---

## 🚀 Setup & Installation

### Create Virtual Environment
```bash
python -m venv .venv
```

### Activate Virtual Environment
```bash
Windows

.venv\Scripts\activate
```
```bash
Linux / macOS

source .venv/bin/activate
```
## ⚠ Disclaimer

MediCare-AI is intended only for educational and research purposes.
It does not provide medical diagnosis and must not be used as a substitute for professional medical advice.
