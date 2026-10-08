import os
import sys
import traceback

# Add project root directory to sys.path so modules like services, models, etc. resolve seamlessly
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# NumPy 1.x <-> 2.x unpickling compatibility bridge
try:
    import numpy as np
    if not hasattr(np, "_core") and hasattr(np, "core"):
        sys.modules["numpy._core"] = np.core
        sys.modules["numpy._core.multiarray"] = np.core.multiarray
except Exception:
    pass

try:
    from app import app
    # Export WSGI entrypoint for Vercel Python serverless runtime
    handler = app
except Exception as exc:
    import logging
    logging.exception("Failed to initialize HeatGuard Flask app in Vercel runtime: %s", exc)
    from flask import Flask, jsonify
    app = Flask(__name__)
    handler = app

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def catch_all(path):
        return jsonify({
            "status": "error",
            "message": "HeatGuard AI serverless cold-start initialization error",
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }), 500
