import os
import base64
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import config
from utils.detector import WeldDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("WeldWebApp")

# Initialize Flask App
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

# Initialize Weld Detector
detector = None
try:
    detector = WeldDetector(model_path=config.MODEL_PATH)
except Exception as err:
    logger.error(f"Initialization alert: Model failed to load at startup. {err}")


def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main application UI."""
    model_loaded = detector.loaded if detector else False
    class_names = list(detector.class_names.values()) if (detector and detector.loaded) else []
    device = detector.device.upper() if (detector and detector.loaded) else "N/A"
    
    return render_template(
        'index.html',
        model_loaded=model_loaded,
        model_path=config.MODEL_PATH,
        class_names=class_names,
        device=device,
        default_conf=config.DEFAULT_CONFIDENCE_THRESHOLD
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_status = "READY" if (detector and detector.loaded) else "ERROR"
    return jsonify({
        "status": "healthy" if model_status == "READY" else "degraded",
        "model_loaded": detector.loaded if detector else False,
        "model_status": model_status,
        "model_path": config.MODEL_PATH,
        "device": detector.device.upper() if (detector and detector.loaded) else "UNKNOWN",
        "class_names": detector.class_names if (detector and detector.loaded) else {},
        "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD
    })


@app.route('/detect', methods=['POST'])
def detect_image():
    """Image detection endpoint for uploaded files."""
    if not detector or not detector.loaded:
        return jsonify({
            "success": False,
            "error": "YOLO model is not loaded. Please place best.pt inside models/ directory and restart."
        }), 500

    # 1. Get uploaded file
    file = None
    if 'file' in request.files:
        file = request.files['file']
    elif 'image' in request.files:
        file = request.files['image']

    if not file or file.filename == '':
        return jsonify({"success": False, "error": "Please select a weld image first."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Unsupported file extension. Please upload JPG, JPEG, PNG, or WEBP."
        }), 400

    # 2. Parse confidence threshold parameter
    conf_threshold = config.DEFAULT_CONFIDENCE_THRESHOLD
    if 'confidence' in request.form:
        try:
            conf_threshold = float(request.form['confidence'])
        except ValueError:
            pass

    try:
        # Read image bytes
        image_bytes = file.read()

        # Run inference
        result = detector.predict(image_bytes, conf_threshold=conf_threshold)

        # Save copy to uploads if desired
        filename = secure_filename(file.filename)
        save_path = os.path.join(config.UPLOAD_FOLDER, filename)
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing image upload: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Unable to process image: {str(e)}"}), 500


@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    """Real-time webcam frame detection endpoint."""
    if not detector or not detector.loaded:
        return jsonify({
            "success": False,
            "error": "YOLO model is not loaded."
        }), 500

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No frame image payload provided."}), 400

    img_b64 = data['image']
    conf_threshold = data.get('confidence', config.DEFAULT_CONFIDENCE_THRESHOLD)

    # Strip header if present (e.g. data:image/jpeg;base64,)
    if ',' in img_b64:
        img_b64 = img_b64.split(',', 1)[1]

    try:
        frame_bytes = base64.b64decode(img_b64)
        result = detector.predict(frame_bytes, conf_threshold=float(conf_threshold))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing webcam frame: {e}")
        return jsonify({"success": False, "error": f"Frame processing error: {str(e)}"}), 500


if __name__ == '__main__':
    import os

    print("=" * 60)
    print("      WELD QUALITY INSPECTION — YOLO AI SYSTEM SERVER      ")
    print("============================================================")
    print(f"Model Path: {config.MODEL_PATH}")
    print("============================================================")

    # Render provides the port via environment variable
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running on port: {port}")
    print("============================================================")

    app.run(host='0.0.0.0', port=port, debug=False)