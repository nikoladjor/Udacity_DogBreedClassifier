import os, sys
from pathlib import Path
from flask import Flask, session, url_for, redirect, flash, get_flashed_messages
from . import classifier_session
from flask_bootstrap import Bootstrap
from flask import render_template, request
# Figure out the forms
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, EmailField, SubmitField, PasswordField
from wtforms.validators import DataRequired, EqualTo

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user

from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from .mlsrc.classification import dog_breed_clasifier, get_model
import secrets
import shutil
import pandas as pd
import numpy as np

from datetime import datetime

UPLOAD_FOLDER = Path(__file__).parent / 'static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def get_num_images(db, hash):
    return len(db[db.hash == hash]['original'].unique())

def get_images(db, hash):
    return list(db[db.hash==hash]['original'].unique())


# create and configure the app
app = Flask(__name__, template_folder='templates/', instance_relative_config=True)
# Load default model on app creation
app._NETWORK = "Inception"
app.secret_key = "VERY_BAD_SECRET_KEY"
app.default_model = get_model(network=app._NETWORK, path_to_models=Path(__file__).parent / "mlsrc/saved_models")
# Mimic the database --> will be deprecated soon
app._db = pd.DataFrame()
Bootstrap(app)


app.config.from_mapping(
    SECRET_KEY='dev',
    SQLALCHEMY_DATABASE_URI='sqlite:///users.db',
    UPLOAD_FOLDER=UPLOAD_FOLDER,
)
# Init the database
db = SQLAlchemy(app=app)
migrate = Migrate(app, db)

# Flask Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Create User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), nullable=False)
    email_address = db.Column(db.String(100), nullable=False, unique=True)
    date_created = db.Column(db.DateTime, default=datetime.now)
    password_hash = db.Column(db.String(256))
    images = db.relationship('Image', backref='user', lazy=True)
    
    # password utils
    @property
    def password(self):
        raise AttributeError("Not allowed to read the password!")
    
    @password.setter
    def password(self, pwd):
        self.password_hash = generate_password_hash(pwd)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        return f"User: <{self.username}>, created on <{self.date_created}>"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create Model to store image information
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(128))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    predictions = db.relationship('BreedPrediction', backref='image', lazy=True)
    def __repr__(self) -> str:
        return f"<Image: {self.image_path}>"


# Prediction probability 
class BreedPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    breed = db.Column(db.String(200), default='None')
    probability = db.Column(db.Float, default=100.0)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'))
    def __repr__(self) -> str:
        return f"<Predicted Breed: {self.breed} with {self.probability} probability>"


# Forms

# Create Login Form
class LoginForm(FlaskForm):
	username = StringField("Username", validators=[DataRequired()])
	password = PasswordField("Password", validators=[DataRequired()])
	submit = SubmitField("Submit")

class UserForm(FlaskForm):
    username = StringField(label="Username", validators=[DataRequired()])
    email_address = EmailField(label="Enter email address", validators=[DataRequired()])
    password = PasswordField('Password', validators=[
        DataRequired(), EqualTo('password_confirm', message="Passwords must match!")
    ])
    password_confirm = PasswordField('Confirm Password', validators=[DataRequired()])
    submit = SubmitField("Submit")

class ImageUploadForm(FlaskForm):
    image = FileField(label="Upload Image", validators=[
        FileRequired(),
        FileAllowed(ALLOWED_EXTENSIONS, message='Only png and jpeg!') 
        ])
    submit = SubmitField("Upload")

# ROUTING
@app.errorhandler(403)
def bad_session(e):
    # note that we set the 404 status explicitly
    return render_template('bad_session.html'), 403

# ensure the instance folder exists
try:
    os.makedirs(app.instance_path)
except OSError:
    pass

# On app creation, remove everything from the sessions folder.

# Home
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/cnn_explained')
def training_description():
    return render_template("cnn_explained.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            # Auth the user here
            if check_password_hash(user.password_hash, form.password.data):
                # OK to login
                login_user(user)
                flash(f"User {form.username.data} logged in successfully!")
            else:
                flash("Wrong password!")
        else:
            # No user -- report error
            flash("No user with that username, please try again.")
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    try:
        logout_user()
        flash("User Logged Out!")
        return redirect(url_for('login'))
    except:
        flash("Error during logout")
        return redirect(url_for('home'))
    

@app.route('/create_session', methods=["GET", "POST"])
def create_session():
    if request.method == "POST":
        session['username'] = request.form['username']
        _now = datetime.now()
        session['date'] = _now.strftime("%d_%m_%y_%H_%M")
        session['date_pp'] = _now.strftime("%a-%d-%B-%H:%M")
        session['random_hash'] = secrets.token_urlsafe(8)
        # Store paths to processed files, as well as results from classification
        # create session folder for the current session
        session['upload_folder'] = os.path.join(app.config['UPLOAD_FOLDER'], session['random_hash'])
        os.makedirs(session['upload_folder'])


        return redirect(url_for('process_image_file'))
    else:
        if 'username' in session:
            # There is no real user auth here, just using session for cleaner_interface
            # this means that session is created, no need to create new one...
            print(f"Session already started for user {session['username']} created at {session['date']}")
            return redirect(url_for('process_image_file'))
        else:
            return render_template("/classifier_session/create_session_form.html")


@app.route('/stop_session')
def stop_session():
    # This should stop the current session and delete the images uploaded
    if 'username' in session:
        shutil.rmtree(session['upload_folder'])
    
    session.clear()
    return redirect(url_for('home'))

@app.route('/face_detector', methods = ['GET', 'POST'])
def process_image_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        
        # Make sure that there is session activated
        if 'username' in session:
            fpath = os.path.join(session['upload_folder'], filename)
        else:
            return redirect(url_for('bad_session'))

        f.save(fpath)

        p_string, breed, probs_dict = dog_breed_clasifier(model=app.default_model, img_path=Path(fpath), network=app._NETWORK)
        chart_labels = list(probs_dict.keys())
        chart_data = list(probs_dict.values())

        tmp_session_header = {
            'filepath': fpath,
            'original': f.filename,
            'predicted_breed': breed,
            'res_string': p_string,
            'hash': session['random_hash']
        }
        
        tmp_session_data = {
            'chart_labels': chart_labels,
            'chart_data': chart_data,
        }

        tmp_df = pd.DataFrame(tmp_session_data)
        for kk,vv in tmp_session_header.items():
            tmp_df[kk] = vv

        if len(app._db) == 0:
            app._db = pd.concat([app._db, tmp_df])
        
        else:
            if not np.isin(f.filename, app._db['original'].unique()):
                app._db = pd.concat([app._db, tmp_df])
        # BUG: FIX HOW DATA ARE STORED!
        selection = app._db[app._db['hash'] == session['random_hash']]
        num_imgs = get_num_images(selection, session['random_hash'])
        imgs = get_images(selection, hash=session['random_hash'])
        
        _chart_lbls = []
        _chart_data = []
        
        for i in range(num_imgs):
            _chart_lbls.append(list(selection[selection['original']==imgs[i]]['chart_labels'].values))
            _chart_data.append(list(selection[selection['original']==imgs[i]]['chart_data'].values))


        _canvas_ids = [f"canvas_{i}" for i in range(num_imgs)]

        # print(app._db['hash'].unique())

        return render_template("/classifier_session/show_result.html", 
                                num_imgs=num_imgs,
                                imgs=imgs,
                                selection=selection,
                                chart_labels = _chart_lbls,
                                chart_data = _chart_data,
                                canvas_array = _canvas_ids,
                                id_list = list(range(len(_canvas_ids)))
                                )
        
    return render_template("/classifier_session/show_result.html", num_imgs=0, canvas_array=[], chart_labels=[], chart_data=[])

@app.route('/user/add', methods=['GET', 'POST'])
def add_user():
    username = None
    form = UserForm()
    # Validate the form
    if form.validate_on_submit():
        usr_query = User.query.filter_by(email_address=form.email_address.data).first()
        if not usr_query:
            # User not found, create one
            user = User(username=form.username.data, email_address=form.email_address.data, password=form.password.data)
            db.session.add(user)
            db.session.commit()

        username = form.username.data
        form.username.data = ''
        form.email_address.data = ''
        flash("User Added!")
    
    all_users = User.query.order_by(User.date_created)
    return render_template("add_user.html", username=username, form=form, all_users=all_users)

@app.route('/user/update/<int:id>')
def update_user(id):
    return "This should update user"

@app.route('/user/delete/<int:id>')
def delete_user(id):
    user_to_delete = User.query.get_or_404(id)
    form = UserForm()
    name = None
    try:
        db.session.delete(user_to_delete)
        db.session.commit()
        flash("user deleted")
        all_users = User.query.order_by(User.date_created)
        # return render_template("add_user.html", name=name, form=form, all_users=all_users)
        return redirect(url_for('add_user', name=name, form=form, all_users=all_users))

    except:
        flash("ERROR! There was a problem deleting user")
        return render_template("add_user.html", name=name, form=form, all_users=all_users)


@app.route('/file/upload', methods=['GET', 'POST'])
@login_required
def file_uploader():
    f = None
    form = ImageUploadForm()
    filename=None
    # Validate the form
    if form.validate_on_submit():
        f = form.image.data
        filename = secure_filename(f.filename)
        form.image.data = None
    
    return render_template("upload_image_test.html", f=f, form=form, filename=filename)


# app.run()