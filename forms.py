from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, DateField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from models import User


# --------------------------------------------------
# Patient Signup Form
# --------------------------------------------------
class PatientSignupForm(FlaskForm):
    full_name = StringField(
        "Full Name",
        validators=[DataRequired(), Length(min=3, max=255)]
    )

    email = StringField(
        "Email",
        validators=[DataRequired(), Email()]
    )

    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=8)]
    )

    confirm_password = PasswordField(
        "Confirm Password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Passwords must match")
        ]
    )

    date_of_birth = DateField(
        "Date of Birth",
        format="%Y-%m-%d",
        validators=[DataRequired()]
    )

    gender = SelectField(
        "Gender",
        choices=[
            ("male", "Male"),
            ("female", "Female"),
            ("other", "Other")
        ],
        validators=[DataRequired()]
    )

    submit = SubmitField("Create Patient Account")

    # --------------------------------------------------
    # Global email uniqueness check (Option A)
    # --------------------------------------------------
    def validate_email(self, email):
        existing_user = User.query.filter_by(email=email.data.lower()).first()
        if existing_user:
            raise ValidationError(
                "An account with this email already exists."
            )


# --------------------------------------------------
# Login Form (Shared for Patient / Doctor)
# --------------------------------------------------
class LoginForm(FlaskForm):
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()]
    )

    password = PasswordField(
        "Password",
        validators=[DataRequired()]
    )

    submit = SubmitField("Login")
