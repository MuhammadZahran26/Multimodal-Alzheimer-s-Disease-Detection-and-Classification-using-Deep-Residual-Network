from datetime import datetime
from extensions import db
from flask_login import UserMixin


# ---------------------------
# Users table
# ---------------------------
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # doctor / patient
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    doctor = db.relationship("Doctor", backref="user", uselist=False)
    patient = db.relationship("Patient", backref="user", uselist=False)


# ---------------------------
# Doctors table
# ---------------------------
class Doctor(db.Model):
    __tablename__ = "doctors"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    full_name = db.Column(db.String(255), nullable=False)
    specialization = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------
# Patients table
# ---------------------------
class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    full_name = db.Column(db.String(255), nullable=False)
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    scan_results = db.relationship("ScanResult", backref="patient", lazy=True)


# ---------------------------
# Scan Results table
# ---------------------------
# ---------------------------
# Scan Results table (Updated)
# ---------------------------
class ScanResult(db.Model):
    __tablename__ = "scan_results"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    # 1. ADDED THIS COLUMN
    file_path = db.Column(db.String(200), nullable=False) 
    prediction = db.Column(db.String(10), nullable=False)  # AD / MCI / CN
    # 2. Note the name here is 'confidence_score'
    confidence_score = db.Column(db.Float, nullable=False)
    scan_type = db.Column(db.String(10))  # MRI / PET
    created_at = db.Column(db.DateTime, default=datetime.utcnow)