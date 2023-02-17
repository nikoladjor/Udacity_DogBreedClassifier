import os
from flask import Flask, json, session, url_for, redirect
from . import classifier_session
from flask_bootstrap import Bootstrap
from flask import render_template, request
from werkzeug.utils import secure_filename
from pathlib import Path
from datetime import datetime
# from .mlsrc.image_processing import face_detector
from .mlsrc.classification import dog_breed_clasifier, get_model
import secrets
import shutil
import pandas as pd
import numpy as np

UPLOAD_FOLDER = Path(__file__).parent / 'static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def get_num_images(db, hash):
    return len(db[db.hash == hash]['original'].unique())

def get_images(db, hash):
    return list(db[db.hash==hash]['original'].unique())


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, template_folder='templates/', instance_relative_config=True)
    # Load default model on app creation
    app._NETWORK = "Inception"
    app.secret_key = "VERY_BAD_SECRET_KEY"
    app.default_model = get_model(network=app._NETWORK, path_to_models=Path(__file__).parent / "mlsrc/saved_models")
    # Mimic the database
    app.db = pd.DataFrame()
    Bootstrap(app)


    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        UPLOAD_FOLDER=UPLOAD_FOLDER,
    )

    @app.errorhandler(403)
    def bad_session(e):
        # note that we set the 404 status explicitly
        return render_template('bad_session.html'), 403
    

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

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
    # a simple page that says hello
    @app.route('/cnn_explained')
    def training_description():
        return render_template("cnn_explained.html")
    
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

            if len(app.db) == 0:
                app.db = pd.concat([app.db, tmp_df])
            
            else:
                if not np.isin(f.filename, app.db['original'].unique()):
                    app.db = pd.concat([app.db, tmp_df])
            # BUG: FIX HOW DATA ARE STORED!
            selection = app.db[app.db['hash'] == session['random_hash']]
            num_imgs = get_num_images(selection, session['random_hash'])
            imgs = get_images(selection, hash=session['random_hash'])
            
            _chart_lbls = []
            _chart_data = []
            
            for i in range(num_imgs):
                _chart_lbls.append(list(selection[selection['original']==imgs[i]]['chart_labels'].values))
                _chart_data.append(list(selection[selection['original']==imgs[i]]['chart_data'].values))


            _canvas_ids = [f"canvas_{i}" for i in range(num_imgs)]

            # print(app.db['hash'].unique())

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


    app.register_blueprint(classifier_session.bp)

    return app

app = create_app()
# app.run()