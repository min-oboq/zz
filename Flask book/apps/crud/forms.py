from flask_wtf import FlaskForm
from wtforms import PasswoedField, StringField, SubmitField
from wtforms.validators import DataRequired, Email, Length

class UserForm(FlaskForm):
    username = StringField(
        "사용자명",
        Validators=[
            DataRequired(message="사용자명은 필수입니다. "),
            Length(max=30, message="30문자 이내로 입력해 주세요. "),
        ],
    )
    
    emali = StringField(
        "메일 주소",
        Validators=[
            DataRequired(message="메일 주소는 필수입니다. "),
            Length(max=30, message="메일 주소의 형식으로 입력해 주세요. "),
        ],
    )
    
    password = PasswoedField(
        "비밀번호",
        Validators=[DataRequired(message="비밀번호는 필수입니다. ")]
    )
    
    
    subit = SubmitField("신규 등록")