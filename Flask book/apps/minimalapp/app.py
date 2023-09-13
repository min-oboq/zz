import logging
import os

from email_validator import validate_email, EmailNotValidError

from flask import Flask, current_app, g, redirect, render_template, request, url_for, flash

from flask_debugtoolbar import DebugToolbarExtension

app.confing["DEBUG_TB_INTERCEPT_REDIRECTS"] = False
toolbar = DebugToolbarExtension(app)

from flask_mail import Mail

app = Flask(__name__)

app.config["SECRET_KEY"] = 'sdlkfsdlkjfs'

app.logger.setLevel(logging.DEBUG)

@app.route("/")
def index():
    return "hellow, flaskbook!"


@app.route("/hello/<name>", methods=["GET", "POST"], endpoint="hello-endpoint")
def hello(name):
    return f"Hello, {name}!"


@app.route("/name/<name>")
def show_name(name):
    
    return render_template("index.html", name=name)


@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/contact/complete", methods=["GET", "POST"])
def contact_complete():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        description = request.form["description"]
        
        is_valid = True
        
        if not username:
            flash("사용자명은 필수입니다")
            is_valid = False
            
            if not email:
                flash("메일 주소는 필수입니다")
                is_valid = False
                
            try:
                validate_email(email)
            except EmailNotValidError:
                flash("메일 주소의 향식으로 입력해 주세요")
                is_valid = False
                
            if not description:
                flash("문의 내용은 필수 입니다")
                is_valid = False
                
            if not is_valid:
                return redirect(url_for("contact"))
            
            flash("문의해 주셔서 감사합니다")
            return redirect(url_for("contact_complete"))
        
        return render_template("contact_complete.html")