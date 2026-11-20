"""Start the development server: python run.py (set FLASK_DEBUG=1 for debug mode)."""
import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
