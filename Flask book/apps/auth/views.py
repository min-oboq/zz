from apps.app import db
from flask import Blueprint, render_template
from apps.auth.forms import SignUpForm, LoginForm
from apps.crud.models import User
from flask import Blueprint, render_template, flash, url_for, redirect, request
from flask_login import login_user, logout_user

auth = Blueprint(
    "auth",
    __name__,
    template_folder="temolates",
    static_folder="static"
)


@auth.route("/logout")
def logout():
    logout_user()
    return render_template(url_for("auth.login"))

    form = SignUpForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        
        if user is not None and user.verify_password(form.password.data):
            login_user(user)
            return redirect(url_for("crud.users"))
        
        flash("메일 주소 또는 비밀전호가 일치하지 않습니다.")
        return render_template("auth/login.html", form=form)
        
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        
        next_ = request.args.get("next")
        if next_ is None or not next_.startswith("/"):
            next_ = url_for("crud.users")
        return redirect(next_)
    
    return render_template("auth/signup.html", form=form)