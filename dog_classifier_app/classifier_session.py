import functools

from flask import (
    Blueprint, render_template, request, session, url_for
)


bp = Blueprint('session', __name__, url_prefix='/session')

# View for running the session

@bp.route('/upload', methods=('GET', 'POST'))
def upload():
    if request.method == 'POST':
        print('POST here....')

    return render_template('classifier_session/upload_image.html')