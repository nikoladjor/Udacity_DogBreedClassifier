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

from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user

from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

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



app.config.from_mapping(
    SECRET_KEY='dev',
    SQLALCHEMY_DATABASE_URI='sqlite:///users.db',
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    FLASK_ADMIN_SWATCH='slate'
)
# Init the database
db = SQLAlchemy(app=app)
migrate = Migrate(app, db)

# Flask Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

admin = Admin(app, name='dogbreed_classifier', template_mode='bootstrap3')

# Create User model
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(200), nullable=False)
    email_address = db.Column(db.String(100), nullable=False, unique=True)
    date_created = db.Column(db.DateTime, default=datetime.now)
    password_hash = db.Column(db.String(256))
    # Hash to create folder on the server --> should 
    folder_hash = db.Column(db.String(8))
    images = db.relationship('Image', backref='user', lazy=True)
    user_upload = db.Column(db.String(100))
    # password utils
    @property
    def password(self):
        raise AttributeError("Not allowed to read the password!")
    
    @password.setter
    def password(self, pwd):
        self.password_hash = generate_password_hash(pwd)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def verify_upload_folder(self):
        # Store paths to processed files, as well as results from classification
        # create session folder for the current session
        if not self.folder_hash:
            self.folder_hash = secrets.token_urlsafe(8)
        self.user_upload = os.path.join(app.config['UPLOAD_FOLDER'], f"usr_{str(self.id)}_{self.folder_hash}")
        os.makedirs(self.user_upload, exist_ok=True)


    def __repr__(self) -> str:
        return f"User: <{self.username}>, created on <{self.date_created}>"

admin.add_view(ModelView(User, db.session))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create Model to store image information
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(128))
    filename = db.Column(db.String(128))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    predictions = db.relationship('BreedPrediction', backref='image', lazy=True)
    
    def __repr__(self) -> str:
        return f"<Image: {self.image_path}>"
    
    @property
    def breed_probabilities(self):
        return BreedPrediction.DictFromPredictions(image_id=self.id)
admin.add_view(ModelView(Image, db.session))


# Prediction probability 
class BreedPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    breed = db.Column(db.String(200), default='None')
    probability = db.Column(db.Float, default=100.0)
    image_id = db.Column(db.Integer, db.ForeignKey('image.id'))
    def __repr__(self) -> str:
        return f"<Predicted Breed: {self.breed} with {self.probability} probability>"

    @classmethod
    def PredictionsFromDict(cls, dict_pairs, image_id):
        for kk, vv in dict_pairs.items():
            try:
                prediction = BreedPrediction(breed=kk, probability=vv, image_id=image_id)
                db.add(prediction)
                db.commit()
            except:
                flash("Prediction not created!")

    @classmethod
    def DictFromPredictions(cls, image_id):
        predictions = BreedPrediction.query.filter_by(image_id=image_id).sort_by(BreedPrediction.probability)
        return predictions
admin.add_view(ModelView(BreedPrediction, db.session))


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
    

@app.route('/user/add', methods=['GET', 'POST'])
def add_user():
    username = None
    form = UserForm()
    # Validate the form
    if form.validate_on_submit():
        usr_query = User.query.filter_by(email_address=form.email_address.data).first()
        if not usr_query:
            # User not found, create one
            user = User(username=form.username.data, 
                        email_address=form.email_address.data, 
                        password=form.password.data)
            user.verify_upload_folder()
            db.session.add(user)
            db.session.commit()

        username = form.username.data
        form.username.data = ''
        form.email_address.data = ''
        flash(f"User Added! Folder: {user.user_upload}")
    
    all_users = User.query.order_by(User.date_created)
    return render_template("add_user.html", username=username, form=form, all_users=all_users)


# TODO: handle the user deletions
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
        # delete folder here
        shutil.rmtree(user_to_delete.user_upload)
        return redirect(url_for('add_user', name=name, form=form, all_users=all_users))

    except:
        flash("ERROR! There was a problem deleting user")
        return render_template("add_user.html", name=name, form=form, all_users=all_users)


@app.route('/user/upload', methods=['GET', 'POST'])
@login_required
def file_uploader():
    f = None
    form = ImageUploadForm()
    filename=None
    # user = User.query.get_or_404(id)

    # Validate the form
    if form.validate_on_submit():
        f = form.image.data
        filename = secure_filename(f.filename)
        form.image.data = None
        fpath = os.path.join(current_user.user_upload, filename)
        try:
            f.save(fpath)
            # After image upload, store the data in database
            img = Image(image_path=fpath, user_id=current_user.id, filename=filename)
            db.session.add(img)
            db.session.commit()
        except:
            raise BufferError("something wrong with the file")
        
        # The image should be commited to the database at this point.
        # Fetch the id and store the predictions
        current_image = Image.query.filter_by(image_path=fpath).first()
        p_string, breed, probs_dict = dog_breed_clasifier(model=app.default_model, img_path=Path(fpath), network=app._NETWORK)
        for kk, vv in probs_dict.items():
            try:
                prediction = BreedPrediction(breed=kk, probability=vv, image_id=current_image.id)
                db.session.add(prediction)
                db.session.commit()
            except:
                flash("ERROR IN STORING PREDICTIONS!")



        
        # Create predictions and commit to the db
        BreedPrediction.PredictionsFromDict(probs_dict, image_id=img.id)
        # return probs_dict
        # TODO: Build the predictions db tables and attach to the image

        
    user_images = Image.query.filter_by(user_id=current_user.id)
    return render_template("upload_image_test.html", f=f, form=form, filename=filename, user=current_user, user_images=user_images)

@app.route('/image/delete/<int:id>')
@login_required
def delete_image(id):
    """Delete image from database

    Args:
        id (int): ID of image to be deleted
    """
    f = None
    form = ImageUploadForm()
    filename=None

    image_to_delete = Image.query.get(id)
    predictions_to_delete = BreedPrediction.query.filter_by(image_id=image_to_delete.id)
    try:
        
        for pred in predictions_to_delete:
            db.session.delete(pred)
            db.session.commit()

        db.session.delete(image_to_delete)
        db.session.commit()
        flash("Image deleted!")
        # delete folder here
        shutil.rmtree(image_to_delete.image_path)
        user_images = Image.query.filter_by(user_id=current_user.id)
        return redirect(url_for('file_uploader', f=f, form=form, filename=filename, user=current_user, user_images=user_images))

    except:
        flash("ERROR! There was a problem deleting user")
    
        user_images = Image.query.filter_by(user_id=current_user.id)
        return render_template("upload_image_test.html", f=f, form=form, filename=filename, user=current_user, user_images=user_images)





# app.run()