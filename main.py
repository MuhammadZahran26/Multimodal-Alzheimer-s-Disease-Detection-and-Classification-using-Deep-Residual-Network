from flask import Flask, request, redirect, url_for, flash, render_template, abort, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from functools import wraps
from extensions import db
from models import User, Doctor, Patient, ScanResult
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
# --------------------------------------------------
# --- NEW IMPORT: The Medical Chatbot ---
from medibot import MedicalChatbot

# --- NEW IMPORT: The AI Brain ---
import preprocessing
# --------------------------------------------------
# App config
# --------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "instance", "app.db")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "dev-secret-key"

app.secret_key = "change_this_to_something_random_secret" 

# --------------------------------------------------
# Init DB
# --------------------------------------------------
db.init_app(app)
# --------------------------------------------------
# --------------------------------------------------
# Init Chatbot Engine
# --------------------------------------------------
meddibot = None
try:
    print(" >>> SYSTEM: Initializing MediBot AI...")
    meddibot = MedicalChatbot()
    print(" >>> SYSTEM: MediBot Ready!")
except Exception as e:
    print(f" >>> WARNING: Chatbot failed to start. Error: {e}")
# --------------------------------------------------
# Flask-Login setup
# --------------------------------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "select_role"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --------------------------------------------------
# 1. LOAD AI MODEL (Global Load)
# --------------------------------------------------
# We load this ONCE when the app starts to save time on every request.
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "final_fusion_model.pth")

# Download if not exists
if not os.path.exists(MODEL_PATH):
    print(" >>> SYSTEM: Downloading model from Hugging Face...")

    MODEL_PATH = hf_hub_download(
        repo_id="MuhammadZahran12/alzheimers-model",  # <-- repo_id
        filename="final_fusion_model.pth",
        local_dir=MODEL_DIR
    )

# Load model
if os.path.exists(MODEL_PATH):
    print(" >>> SYSTEM: Found Model File. Loading AI...")
    ai_model = preprocessing.load_model(MODEL_PATH)
else:
    print(" >>> WARNING: Model file not found.")
    ai_model = None

# --------------------------------------------------
# Create tables (safe)
# --------------------------------------------------
with app.app_context():
    os.makedirs(os.path.join(BASE_DIR, "instance"), exist_ok=True)
    db.create_all()

# -- Reusable Role Guard
def role_required(required_role):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                flash("Please log in first.", "error")
                return redirect(url_for("select_role"))

            if current_user.role != required_role:
                flash("You are not authorized to access this page.", "error")

                # redirect user to their correct dashboard
                if current_user.role == "doctor":
                    return redirect(url_for("doctor_dashboard"))
                else:
                    return redirect(url_for("patient_dashboard"))

            return view_func(*args, **kwargs)
        return wrapper
    return decorator

# --------------------------------------------------
# --- THE SCRAPER ENGINE ---
def fetch_medical_info(topic_key):
    # 1. Fallback Data (The "Safety Net" for your FYP Presentation)
    # If scraping fails, the app will show this instead of an error.
    fallback_content = {
        'causes': {
            'title': 'Why Alzheimer\'s Occurs?',
            'body': """
                <h3>The Biological Basis</h3>
                <p>Alzheimer's disease is thought to be caused by the abnormal build-up of proteins in and around brain cells. One of the proteins involved is called amyloid, deposits of which form plaques around brain cells. The other protein is called tau, deposits of which form tangles within brain cells.</p>
                <h3>Neurotransmitters</h3>
                <p>Although it's not known exactly what causes this process to begin, scientists now know that it begins many years before symptoms appear. As brain cells become affected, there's also a decrease in chemical messengers (called neurotransmitters) involved in sending messages, or signals, between brain cells.</p>
            """,
            'source_url': 'https://www.nia.nih.gov/'
        },
        'diagnosis': {
            'title': 'MRI & PET Scan Examination',
            'body': """
                <h3>Magnetic Resonance Imaging (MRI)</h3>
                <p>MRI scans use powerful radio waves and magnets to create a detailed view of your brain. In Alzheimer's detection, doctors look for shrinkage in specific areas like the hippocampus and entorhinal cortex, which are responsible for memory.</p>
                <h3>PET Scans</h3>
                <p>Positron emission tomography (PET) scans use a radioactive tracer substance to look for disease. Newer PET scans can detect the amyloid plaques and tau tangles associated with Alzheimer's disease.</p>
            """,
            'source_url': 'https://www.mayoclinic.org/'
        },
        'age-risk': {
            'title': 'Risk Factors & Prevention',
            'body': """
                <h3>Age: The Primary Factor</h3>
                <p>Increasing age is the greatest known risk factor for Alzheimer's disease. It is not a part of normal aging, but the likelihood of developing it increases dramatically after age 65. One study found that there are four new diagnoses per 1,000 people ages 65 to 74.</p>
                
                <h3>Mild Cognitive Impairment (MCI)</h3>
                <p>MCI is a decline in memory or other thinking skills that is greater than normal for a person's age, but does not prevent them from functioning in social or work environments. People who have MCI have a significant risk of developing dementia. This is a critical stage for early detection.</p>
                
                <h3>Family History and Genetics</h3>
                <p>Your risk of developing Alzheimer's is somewhat higher if a first-degree relative—your parent, brother or sister—has the disease. Most genetic mechanisms of Alzheimer's among families remain unexplained, and the genetic factors are likely complex.</p>
            """,
            'source_url': 'https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447'
        }
    }

    # 2. Live Sources
    sources = {
        'causes': 'https://www.nia.nih.gov/health/alzheimers-causes-and-risk-factors/what-causes-alzheimers-disease',
        'diagnosis': 'https://www.nia.nih.gov/health/alzheimers-symptoms-and-diagnosis/how-alzheimers-disease-diagnosed',
        'age-risk': 'https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447'
    }

    url = sources.get(topic_key)
    if not url:
        return None

    print(f"DEBUG: Attempting to scrape {url}...") # Watch your terminal for this

    try:
        # Use a "Real" Browser Header to avoid 403 Forbidden errors
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5)
        
        print(f"DEBUG: Status Code: {response.status_code}") # 200 is good, 403 is blocked

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Smart Selector: Try multiple common content areas
            # "article" is standard HTML5, "main" is standard, "role=main" is accessible
            content_div = (
                soup.find('article') or 
                soup.find('main') or 
                soup.find('div', role='main') or
                soup.select_one('.node__content') # Old specific one
            )

            if content_div:
                # Cleanup: Remove images, scripts, navs inside the content
                for tag in content_div.find_all(['img', 'script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()

                return {
                    'title': fallback_content[topic_key]['title'], # Keep our clean title
                    'body': str(content_div),
                    'source_url': url
                }
    except Exception as e:
        print(f"DEBUG: Scraping failed due to: {e}")

    # 3. IF SCRAPING FAILS -> RETURN FALLBACK (Crucial for FYP)
    print("DEBUG: Using Fallback Data")
    return fallback_content.get(topic_key)

# --------------------------------------------------

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def home():
    return redirect(url_for("home_page"))

# ---------- SIGNUP (PATIENT) ----------
@app.route("/signup/patient", methods=["POST"])
def patient_signup():
    username = request.form["username"]
    email = request.form["email"]
    password = request.form["password"]
    full_name = request.form["full_name"]

    # Check if user already exists
    # check duplicate username
    existing_username = User.query.filter_by(username=username).first()
    if existing_username:
        flash("Username already exists. Please choose another.", "error")
        return redirect(url_for("patient_signup"))

    # check duplicate email
    existing_email = User.query.filter_by(email=email).first()
    if existing_email:
        flash("Email already exists. Please use another email.", "error")
        return redirect(url_for("patient_signup"))
    
    if existing_username and existing_email:
        flash("Username and Email already exist. Please choose another.", "error")
        return redirect(url_for("patient_signup"))


    # 1. Create User (Login Info)
    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password),
        role="patient"
    )
    db.session.add(user)
    db.session.commit()

    # 2. Create Patient (Medical Info) linked to User
    patient = Patient(
        user_id=user.id,
        full_name=full_name
    )
    db.session.add(patient)
    db.session.commit()

    # 3. SUCCESS: Notify user and move to Login page
    flash("Account created successfully! Please login.", "success")
    return redirect(url_for('patient_login_page'))

# ---------- SIGNUP (DOCTOR) ----------
@app.route("/signup/doctor", methods=["POST"])
def doctor_signup():
    # build full_name from form
    first_name = request.form["first_name"].strip()
    last_name = request.form["last_name"].strip()
    full_name = f"{first_name} {last_name}"

    username = request.form["username"].strip()
    email = request.form["email"].strip().lower()
    password = request.form["password"]

    # duplicate checks (same as patient)
    if User.query.filter_by(username=username).first():
        flash("Username already exists. Please choose another.", "error")
        return redirect(url_for("doctor_signup_page"))

    if User.query.filter_by(email=email).first():
        flash("Email already exists. Please use another email.", "error")
        return redirect(url_for("doctor_signup_page"))
    if User.query.filter_by(username=username).first() and User.query.filter_by(email=email).first():
        flash("Username and Email already exist. Please choose another.", "error")
        return redirect(url_for("doctor_signup_page"))
    
    # create user
    user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password),
        role="doctor"
    )
    db.session.add(user)
    db.session.commit()

    # create doctor profile
    doctor = Doctor(
        user_id=user.id,
        full_name=full_name
    )
    db.session.add(doctor)
    db.session.commit()

    flash("Doctor account created successfully. Please login.", "success")
    return redirect(url_for("doctor_login_page"))


# ---------- LOGIN (PATIENT AND DOCTOR LOGIC) ----------
@app.route("/login/patient", methods=["POST"])
def patient_login():
    email = request.form["email"].strip().lower()
    password = request.form["password"]
    
    # 1. NEW: Check if the 'remember' box was clicked
    remember = True if request.form.get("remember") else False
    
    user = User.query.filter_by(email=email, role="patient").first()

    if not user or not check_password_hash(user.password_hash, password):
        flash("Patient account not found or invalid credentials.", "error")
        return redirect(url_for("patient_login_page"))
    
    # 2. NEW: Pass 'remember=True' to the login function
    login_user(user, remember=remember)

    login_user(user)
    return redirect(url_for("patient_dashboard"))

@app.route("/login/doctor", methods=["POST"])
def doctor_login():
    email = request.form["email"].strip().lower()
    password = request.form["password"]

    # 1. NEW: Check for doctor as well
    remember = True if request.form.get("remember") else False

    user = User.query.filter_by(email=email, role="doctor").first()

    if not user or not check_password_hash(user.password_hash, password):
        flash("Doctor account not found or invalid credentials.", "error")
        return redirect(url_for("doctor_login_page"))
    
    # 2. NEW: Pass remember here too
    login_user(user, remember=remember)

    login_user(user)
    return redirect(url_for("doctor_dashboard"))


# ---------- LOGOUT ----------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("select_role"))


# --------------------------------------------------
#  DOCTOR: VIEW SINGLE PATIENT RECORDS (MISSING ROUTE FIX)
# --------------------------------------------------
@app.route("/doctor/patient/<int:patient_id>")
@login_required
@role_required("doctor")
def patient_record(patient_id):
    # 1. Fetch patient
    patient = Patient.query.get_or_404(patient_id)
    
    # 2. Fetch ALL scans for this patient (History)
    scans = ScanResult.query.filter_by(patient_id=patient.id)\
        .order_by(ScanResult.created_at.desc()).all()
        
    return render_template("doctor_patient_view.html", patient=patient, scans=scans, user=current_user)
@app.route("/patient/dashboard")
@login_required
@role_required("patient")
def patient_dashboard():
    # 1. Find the Patient profile linked to the current User
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    
    # 2. Get the scans for this patient (newest first)
    scans = []
    if patient:
        scans = ScanResult.query.filter_by(patient_id=patient.id).order_by(ScanResult.created_at.desc()).all()
    
    # 3. Pass the 'scans' list to the HTML
    return render_template("patient_dashboard.html", user=current_user, scans=scans, patient=patient)

# --------------------------------------------------
#  AI PROCESSING ROUTE (The Core Logic)
# --------------------------------------------------
@app.route("/upload/process", methods=["POST"])
@login_required
@role_required("patient")
def patient_upload():
    # A. Get Files
    mri_file = request.files.get("mri_scan")
    pet_file = request.files.get("pet_scan")

    # B. Get Demographics (Required for Fusion Model)
    age = request.form.get("patient_age")
    gender = request.form.get("patient_gender")

    if not mri_file and not pet_file:
        flash("Please upload at least one scan.", "error")
        return redirect(url_for("upload_process"))

    # C. Find Patient Profile
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    if not patient:
        flash("Patient profile error.", "error")
        return redirect(url_for("patient_dashboard"))

    # D. Save Files to Disk
    mri_path = None
    pet_path = None

    if mri_file and mri_file.filename:
        filename = secure_filename(mri_file.filename)
        mri_path = os.path.join(UPLOAD_FOLDER, f"MRI_{filename}")
        mri_file.save(mri_path)

    if pet_file and pet_file.filename:
        filename = secure_filename(pet_file.filename)
        pet_path = os.path.join(UPLOAD_FOLDER, f"PET_{filename}")
        pet_file.save(pet_path)

    # E. RUN AI PREDICTION
    pred_class = "PENDING"
    confidence = 0.0
    
    if ai_model:
        try:
            print(f"Running Fusion Model on MRI: {mri_path} | PET: {pet_path} | Age: {age}")
            # Call our preprocessing module
            result = preprocessing.get_prediction(ai_model, mri_path, pet_path, age, gender)
            
            pred_class = result['prediction']
            confidence = result['confidence']
            print(f"AI Result: {pred_class} ({confidence}%)")
            
        except Exception as e:
            print(f"AI Error: {e}")
            flash("AI Analysis failed, but file saved.", "error")

    # F. Save Result to Database
    # We create ONE 'Fusion' record for this upload session
    # ... (previous code where you get pred_class and confidence) ...
    scan_type = "FUSION"
    if mri_path and not pet_path: scan_type = "MRI"
    if pet_path and not mri_path: scan_type = "PET"
    
    # Use MRI path as primary, or PET if MRI missing
    # We save the RELATIVE path (e.g., 'uploads/filename.nii') not the absolute path
    if mri_file:
        final_filename = secure_filename(mri_file.filename)
    else:
        final_filename = secure_filename(pet_file.filename)
        
    # We store relative path for HTML to use later
    db_path = f"uploads/{final_filename}" 

    new_scan = ScanResult(
        patient_id=patient.id,
        scan_type=scan_type,
        
        # 1. Matches the new column in models.py
        file_path=db_path, 
        
        prediction=pred_class,
        
        # 2. Changed 'confidence' to 'confidence_score' to match models.py
        confidence_score=confidence  
    )
    
    db.session.add(new_scan)
    db.session.commit()

    flash("Analysis Complete!", "success")
    # Redirect to the NEW Result Page
    return redirect(url_for("view_result", scan_id=new_scan.id))

# Upload process route (placeholder)
@app.route("/upload/process")
@login_required
def upload_process():
    if current_user.role != "patient":
        return {"error": "Access denied"}, 403
    return render_template("upload_process.html", user=current_user)

# --------------------------------------------------
# 3. RESULT VIEW ROUTE (Displays the Report)
# --------------------------------------------------
@app.route("/results/<int:scan_id>")
@login_required
def view_result(scan_id):
    scan = ScanResult.query.get_or_404(scan_id)
    
    # Security: Ensure patient owns this scan
    if current_user.role == "patient":
        patient = Patient.query.filter_by(user_id=current_user.id).first()
        if scan.patient_id != patient.id:
            abort(403)
            
    # If doctor, verify they are allowed (Simplified for now: Doctors can view all)
    
    return render_template("results.html", scan=scan)

# --------------------------------------------------
#  DOCTOR DASHBOARD ROUTE (The missing piece!)
# --------------------------------------------------
@app.route("/doctor/dashboard")
@login_required
@role_required("doctor")
def doctor_dashboard():
    # 1. Fetch all patients
    all_patients = Patient.query.all()
    
    patient_data = []
    
    for p in all_patients:
        # 2. Get the LATEST scan for this patient
        last_scan = ScanResult.query.filter_by(patient_id=p.id)\
            .order_by(ScanResult.created_at.desc()).first()
        
        # 3. Build the data object expected by the HTML
        status = "Pending"
        confidence = 0.0
        last_date = "No scans"
        
        if last_scan:
            status = last_scan.prediction  # AD, MCI, or CN
            # Handle percentage logic (if saved as 98.5 or 0.985)
            confidence = last_scan.confidence_score
            if confidence > 1: confidence = confidence / 100.0
            
            last_date = last_scan.created_at.strftime("%b %d")
            
        patient_data.append({
            'id': p.id,
            'name': p.full_name,
            'status': status,
            'confidence': confidence,
            'last_scan': last_date
        })
    
    return render_template("doctor_dashboard.html", user=current_user, patients=patient_data)

# ---------- DOCTOR REVIEW ROUTES ----------
@app.route("/doctor/scans")
@login_required
@role_required("doctor")
def doctor_scans():
    scans = (
        db.session.query(ScanResult, Patient)
        .join(Patient, ScanResult.patient_id == Patient.id)
        .order_by(ScanResult.created_at.desc())
        .all()
    )

    return render_template("doctor_scans.html", scans=scans)

# --------------------------------------------------
# -------------------------
# Page routes (HTML)
# -------------------------

@app.route("/home")
def home_page():
    return render_template("home.html")

@app.route("/select-role")
def select_role():
    return render_template("select_role.html")

@app.route("/login/patient")
def patient_login_page():
    return render_template("patient_login.html")

@app.route("/login/doctor")
def doctor_login_page():
    return render_template("doctor_login.html")

@app.route("/signup/patient")
def patient_signup_page():
    return render_template("patient_signup.html")

@app.route("/signup/doctor")
def doctor_signup_page():
    return render_template("doctor_signup.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/learn-more")
def learn_more():
    return render_template("learn_more.html")

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    # Initialize variables to ensure they exist for the template
    result_text = None
    risk_level = None

    if request.method == 'POST':
        # 1. Define the specific questions we are looking for
        # These must match the 'name' attributes in your HTML (q1, q2, q3, q4)
        questions = ['q1', 'q2', 'q3', 'q4']
        score = 0
        
        # 2. Check each specific question
        for q in questions:
            # .get() prevents crashing if a user skips a question
            answer = request.form.get(q) 
            if answer == 'yes':
                score += 1
        
        # 3. Logic: Determine Result
        # If they answered 'yes' to 2 or more questions, we flag as High Risk
        if score >= 2:
            risk_level = "high"
            result_text = (
                f"You answered 'Yes' to {score} out of {len(questions)} indicators. "
                "Your results suggest potential cognitive concerns. "
                "We highly recommend creating an account to use our AI Scan tool."
            )
        else:
            risk_level = "low"
            result_text = (
                f"You answered 'Yes' to {score} out of {len(questions)} indicators. "
                "Your results are currently within a normal range. "
                "However, regular monitoring is the best prevention."
            )
            
        return render_template('quiz.html', result=result_text, risk=risk_level)

    # GET request (just showing the page)
    return render_template('quiz.html', result=None, risk=None)

@app.route("/admin_review")
def admin_review():
    return render_template("admin_review.html")

@app.route('/learn/<topic>')
def learn_more_topic(topic):
    # This route dynamically loads the content based on the box clicked
    content = fetch_medical_info(topic)
    if not content:
        abort(404)
    return render_template('article_view.html', content=content)
# --------------------------------------------------
# --------------------------------------------------
# CHATBOT ROUTE
# --------------------------------------------------
@app.route("/get_response", methods=["POST"])
def get_chat_response():
    if not meddibot:
        return jsonify({"response": "System Error: Chatbot is offline (Check API Key or Vector DB)."})

    user_input = request.form.get("msg")
    session_id = request.form.get("session_id", "default_user") # Simple session handling

    if not user_input:
        return jsonify({"response": "Please say something."})

    try:
        response_text = meddibot.get_response(user_input, session_id)
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"response": "I'm having trouble thinking right now. Please try again."})
# --------------------------------------------------
# Inside main.py

@app.route("/api/global-stats")
def global_stats():
    # Real-world data based on Alzheimer's Disease International (ADI) & WHO projections
    # Source: https://www.alzint.org/resource/world-alzheimer-report-2023/
    data = {
        "years": ["2020", "2023", "2026", "2030", "2040", "2050"],
        "cases": [55, 62, 70, 78, 116, 139]  # Cases in Millions
    }
    return jsonify(data)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
