from flask import render_template, flash, request, url_for, redirect, abort, session, Markup
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
from application import app, bcrypt, mail, login_manager
from application.classes.user import User
from application.forms.forms import RegistrationForm, LoginForm
 
import os 
import json 
import re
from bson import ObjectId

@login_manager.user_loader
def load_user(user_id):
    user_id = str(user_id)
    return User.get_by_id(user_id)

@app.route("/home", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    print(current_user.is_authenticated)
    return render_template("home.html")

@app.route("/session", methods=["GET", "POST"])
def session():
    return render_template("session.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    
    print(form.email.data, form.password.data, form.confirm_password.data, form.validate_on_submit())

    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(id=str(ObjectId()), email=form.email.data, password=hashed_pw, _is_active=True, sessions=[])
        user.add()
        flash('Your account has been created', 'success')
        return redirect('login')

    return render_template("register.html", form=form)

@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.get_by_email(form.email.data)
        print(user)
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            else:
                return redirect(url_for('home'))
        else:
            flash("Login Unsuccessful. Please check email and password", 'danger')
    
    return render_template("login.html", form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))