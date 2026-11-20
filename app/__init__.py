"""Flask web application for ECG ischemia prediction."""
import os

from flask import Flask

from app.config import load_config


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.update(load_config())
    if overrides:
        app.config.update(overrides)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    from app.routes import register_routes
    register_routes(app)
    return app
