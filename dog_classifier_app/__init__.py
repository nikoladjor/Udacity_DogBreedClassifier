import os
from flask import Flask, json
from . import classifier_session
from flask_bootstrap import Bootstrap
from flask import render_template, request
from werkzeug.utils import secure_filename
from pathlib import Path
# from .mlsrc.image_processing import face_detector
from .mlsrc.classification import dog_breed_clasifier, get_model

UPLOAD_FOLDER = Path(__file__).parent / 'static/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, template_folder='templates/', instance_relative_config=True)
    # Load default model on app creation
    app._NETWORK = "Inception"
    app.default_model = get_model(network=app._NETWORK, path_to_models=Path(__file__).parent / "mlsrc/saved_models")
    Bootstrap(app)

    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        UPLOAD_FOLDER=UPLOAD_FOLDER
    )
    

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

    # Home
    @app.route('/')
    def home():
        return render_template("index.html")
    # a simple page that says hello
    
    @app.route('/session')
    def hello():
        return render_template("/classifier_session/upload_image.html")

    
    @app.route('/face_detector', methods = ['GET', 'POST'])
    def process_image_file():
        if request.method == 'POST':
            f = request.files['file']

            filename = secure_filename(f.filename)
            fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            f.save(fpath)
            fname_ed = filename.split('.')[0] + '_ed.png'

            p_string, breed, probs_dict = dog_breed_clasifier(model=app.default_model, img_path=Path(fpath), network=app._NETWORK)
            chart_labels = json.dumps(list(probs_dict.keys()))
            chart_data = json.dumps(list(probs_dict.values()))
            return render_template("/classifier_session/show_result.html", 
                                    original=f.filename, breed=breed, res_string=p_string, chart_labels=chart_labels, chart_data=chart_data)

    app.register_blueprint(classifier_session.bp)
    

    return app