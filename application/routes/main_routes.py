from flask import render_template, flash, request, url_for, redirect, abort, session, Markup
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
from application import app, bcrypt, mail, login_manager
from application.classes.user import User
from application.classes.session import Session
from application.forms.forms import RegistrationForm, LoginForm

from utils import plot_overall_mood, blink_rate_graph, get_blink_rate, get_blink_ratio, current_blink_rate, calc_focus_values, update_focus_value, update_focus_plots, get_mood, get_overall_mood, plot_moods
from io import BytesIO 
from io import StringIO
from PIL import Image

from datetime import datetime
import os 
import json 
import re
import time
import numpy as np
import cv2
from bson import ObjectId
import base64

@login_manager.user_loader
def load_user(user_id):
    user_id = str(user_id)
    return User.get_by_id(user_id)

@app.route("/home", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def home():
    print(current_user.is_authenticated)
    return render_template("home.html")

@app.route("/start-session")
def start_session():
    return render_template("start_session.html")

@app.route("/end-session/<string:session_id>")
def end_session(session_id):
    # current_session = current_user.get_session_by_id(session_id)
    # current_user.update_session(session_id, end_time=datetime.now, total_time=datetime.now() - current_session.start_time)

    return render_template("end_session.html")

@app.route("/session", methods=["GET", "POST"])
def session():
    if current_user.id == "5fd5651341432c9630f95c8e":
        current_session = current_user.get_session_by_id('5fd63ac9921dc84394b08bda')
        session_id = current_session.id
        header = ""
    else:
        current_session = Session(id=str(ObjectId()), start_time=datetime.now())
        current_user.add_session(current_session)
        session_id = current_session.id
        header = "Currently Unavailable for Standard Users, Sample Data Displayed"
    return render_template("session.html", session_id=session_id, header=header)

@login_required
@app.route("/account")
def account():
    user = current_user
    sessions = current_user.sessions
    info = [[session.start_time.strftime('%I:%M')] for session in sessions]
    return render_template("account.html", user=user, sessions=sessions, info=info)

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

@app.route("/focus", methods=["GET", "POST"])
def focus():
    data_url = request.values['image']
    img_64 = data_url.replace("data:image/png;base64,", "")
    png_as_np = np.frombuffer(base64.b64decode(img_64), dtype=np.uint8)
    print(png_as_np.shape)
    image_buffer = cv2.imdecode(png_as_np, cv2.IMREAD_COLOR)
    cv2.imwrite("b64out.png",image_buffer)

    

    # nparr = np.fromstring(img_bytes, np.uint8)
    # print(nparr.shape)
    # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # print("SHAPE", img_np.shape)

    return {"status": "failure"}, 200

    # img = Image.open(BytesIO(img_bytes))
    # img  = np.array(img)
    # print(img.shape)

    # session_id = request.get_json().get("id")
    # session = current_user.get_session_by_id(session_id)
    # img_64 = request.get_json().get("img")
    
    # img_bytes = base64.b64decode(img_64)
    # img = Image.open(BytesIO(img_bytes))
    # img  = np.array(img)
    # print(img)
    # cv2.imshow(img)


    # print(len(img_64))
    # decoded = base64.b64decode(img_64)
    # img_np = np.frombuffer(decoded, np.uint8)
    # print(img_np.shape)
    # img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    # print(img is None)

    # focus, ratio = focus_method(img, session.focus, session.ratios)

    # current_user.update_session(session_id, {"focus": session.focus.append(focus), "ratio": session.ratios.append(ratio)})

    # value = display_focus_value(focus, len(session.focus))

    # print(value)
    # return value

@app.route("/strain")
def strain():
    session_id = request.args.get("id")
    session = current_user.get_session_by_id(session_id)
    img_64 = request.args.get("img")
    img, time = None, 0
    
    blinks = get_blink_rate(img, time, 7, session.blinks)
    blink_rate = current_blink_rate(blinks, 10, time)

    current_user.update_session(session_id, {"blinks": session.blinks, "blink_rate": session.blink_rate.append(blink_rate)})

    print("HERE")
    return "hi"

@app.route("/mood")
def mood():
    session_id = request.args.get("id")
    session = current_user.get_session_by_id(session_id)
    img_64 = request.args.get("img")
    img = img_64.decode('base64')

    mood = get_mood(img)
    current_user.update_session(session_id, {"mood": session.mood.append(mood)})

    print("HERE")
    return "hi"

@app.route("/focus-line/<string:session_id>")
def focus_line(session_id):
    session = current_user.get_session_by_id(session_id)
    line, _ = update_focus_plots(session.focus, session.ratios)
    return line

@app.route("/focus-heat/<string:session_id>")
def focus_heat(session_id):
    session = current_user.get_session_by_id(session_id)
    _, heat = update_focus_plots(session.focus, session.ratios)
    return heat

@app.route("/mood-line/<string:session_id>")
def mood_line(session_id):
    session = current_user.get_session_by_id(session_id)
    line = plot_overall_mood(session.mood)
    return line

@app.route("/mood-pie/<string:session_id>")
def mood_pie(session_id):
    session = current_user.get_session_by_id(session_id)
    pie = plot_moods(session.mood[-1])
    return pie

@app.route("/strain-line/<string:session_id>")
def strain_line(session_id):
    session = current_user.get_session_by_id(session_id)
    line = blink_rate_graph(session.blink_rate, 25)
    return line

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))