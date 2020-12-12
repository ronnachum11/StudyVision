from datetime import datetime
from datetime import date

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextField, TextAreaField, BooleanField, DateField, RadioField, SelectField, PasswordField
from wtforms.fields.html5 import DateField
from wtforms_components import ColorField, TimeField
from wtforms.validators import DataRequired, Email, Length, EqualTo, Regexp, ValidationError
from application.classes.user import User

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')