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

UPLOAD_FOLDER = Path(__file__).parent / 'static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, template_folder='templates/', instance_relative_config=True)
    # Load default model on app creation
    app._NETWORK = "Inception"
    app.secret_key = "VERY_BAD_SECRET_KEY"
    app.default_model = get_model(network=app._NETWORK, path_to_models=Path(__file__).parent / "mlsrc/saved_models")
    Bootstrap(app)

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        UPLOAD_FOLDER=UPLOAD_FOLDER
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
    
    @app.route('/create_session', methods=["GET", "POST"])
    def create_session():
        if request.method == "POST":
            session['username'] = request.form['username']
            _now = datetime.now()
            session['date'] = _now.strftime("%d_%m_%y_%H_%M")
            session['date_pp'] = _now.strftime("%a-%d-%B-%H:%M")
            session['random_hash'] = secrets.token_urlsafe(8)

            # create session folder for the current session
            session['upload_folder'] = os.path.join(app.config['UPLOAD_FOLDER'], session['random_hash'])
            os.makedirs(session['upload_folder'])

            return redirect(url_for('classifier_task'))
        else:
            if 'username' in session:
                # There is no real user auth here, just using session for cleaner_interface
                # this means that session is created, no need to create new one...
                print(f"Session already started for user {session['username']} created at {session['date']}")
                return redirect(url_for('classifier_task'))
            else:
                return render_template("/classifier_session/create_session_form.html")

    @app.route('/session', methods=['GET', 'POST'])
    def classifier_task():
        if 'username' in session:
            return render_template("/classifier_session/upload_image.html")
        else:
            redirect(url_for('create_session'))

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
            filename = secure_filename(f.filename)#
            
            # Make sure that there is session activated
            if 'username' in session:
                fpath = os.path.join(session['upload_folder'], filename)
            else:
                return redirect(url_for('bad_session'))

            f.save(fpath)
            fname_ed = filename.split('.')[0] + '_ed.png'

            p_string, breed, probs_dict = dog_breed_clasifier(model=app.default_model, img_path=Path(fpath), network=app._NETWORK)
            chart_labels = json.dumps(list(probs_dict.keys()))
            chart_data = json.dumps(list(probs_dict.values()))
            return render_template("/classifier_session/show_result.html", 
                                    original=f.filename, breed=breed, res_string=p_string, chart_labels=chart_labels, chart_data=chart_data)





    app.register_blueprint(classifier_session.bp)

    return app