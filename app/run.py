import json
import os
import pickle

from flask import current_app
from views import app

if __name__ == "__main__":
    with app.app_context():
        try:
            model = None
            with open("app/models/knn_model.model", "rb") as knn_model:
                model = pickle.load(knn_model)
            current_app.model = model
            with open("app/models/labels.json") as label:
                labels = json.load(label)
            current_app.labels = labels
        except Exception:
            print("run.py: Gagal memuat model")
            os.abort()

    app.run("0.0.0.0", 80, True)
