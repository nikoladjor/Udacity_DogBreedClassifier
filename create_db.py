import sys
print(sys.path)
sys.path.append("./dog_classifier_app")

from dog_classifier_app.app import app
from dog_classifier_app.app import db


with app.app_context():
    db.create_all()
