import os
import re
import logging
import random
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.conf import settings
from datetime import datetime, timedelta
import requests
from typing import Optional, Dict
import joblib

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATHS = {
    "username_model": BASE_DIR / "profile_checker" / "svm_model.pkl",
    "vectorizer": BASE_DIR / "profile_checker" / "vectorizer.pkl",
    "image_model": BASE_DIR / "profile_checker" / "profile_detector_model.tflite",  # No change if path is same
}


MODEL_CACHE = {
    "svm_model": None,
    "vectorizer": None,
    "image_model": None,
}


# URL Patterns
URL_PATTERNS = {
    "twitter": r"(?:twitter|x)\.com/([^/?#]+)",
    "instagram": r"instagram\.com/([^/?#]+)",
    "facebook": r"facebook\.com/([^/?#]+)",
    "tiktok": r"tiktok\.com/@?([^/?#]+)",
    "linkedin": r"linkedin\.com/in/([^/?#]+)",
}

ADVICE_MESSAGES = {
    "Real Account": [
        "This profile appears to be genuine based on our analysis.",
        "Our system indicates this is likely a real profile. However, always practice caution online.",
        "The profile shows characteristics consistent with legitimate accounts.",
        "This appears to be an authentic profile, but always verify through multiple channels when sharing sensitive information.",
        "Our detection system classifies this as a real profile. Continue with normal online safety practices.",
        "The profile analysis suggests this is a legitimate account. Always exercise standard online precautions."
    ],
    "Fake Account": [
        "This profile shows multiple signs of being fake. Proceed with extreme caution.",
        "Our system has flagged this as a potential fake account. Avoid sharing personal information.",
        "Several indicators suggest this may be an inauthentic profile. Consider blocking and reporting.",
        "Exercise caution - this profile contains characteristics commonly associated with fake accounts.",
        "We recommend not engaging with this profile as it shows patterns consistent with fake accounts.",
        "The analysis suggests this is likely a fake profile. Do not share personal details or financial information.",
        "This profile exhibits suspicious characteristics. Consider verifying the identity through other means before engaging."
    ],
    "default": [
        "Analysis was inconclusive. Proceed with standard online safety practices.",
        "Unable to determine profile authenticity with high confidence. Exercise normal caution.",
        "Our system couldn't make a definitive assessment. Practice standard online safety."
    ]
}

# ------------------------- Model Loaders -------------------------
def load_model(model_key: str, path_key: str) -> bool:
    if MODEL_CACHE[model_key] is None:
        try:
            MODEL_CACHE[model_key] = joblib.load(MODEL_PATHS[path_key])
            logger.info(f"✅ Loaded model: {model_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load {model_key}: {e}", exc_info=True)
            return False
    return True

def load_username_model() -> bool:
    return load_model("svm_model", "username_model") and load_model("vectorizer", "vectorizer")

def load_image_model() -> bool:
    if MODEL_CACHE["image_model"] is None:
        try:
            if not MODEL_PATHS["image_model"].exists():
                logger.error(f"❌ TFLite model file not found at {MODEL_PATHS['image_model']}")
                return False
            interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATHS["image_model"]))
            interpreter.allocate_tensors()
            MODEL_CACHE["image_model"] = interpreter
            logger.info("✅ Loaded TFLite image model")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load TFLite model: {e}", exc_info=True)
            return False
    return True

# ------------------------- Prediction Logic -------------------------
def preprocess_image_for_tflite(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(image_path: str) -> Dict[str, str]:
    try:
        if not load_image_model():
            raise ValueError("Model loading failed.")
        interpreter = MODEL_CACHE["image_model"]

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Image could not be read by OpenCV.")
        img = preprocess_image_for_tflite(img)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0][0]
        result = "Real Account" if output < 0.5 else "Fake Account"
        return {"image_result": result, "advice": random.choice(ADVICE_MESSAGES[result])}
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return None

def extract_username(profile_url: str) -> Optional[str]:
    try:
        if not profile_url:
            return None
        if not profile_url.startswith(('http://', 'https://')):
            profile_url = 'https://' + profile_url
        for pattern in URL_PATTERNS.values():
            match = re.search(pattern, profile_url)
            if match:
                return match.group(1)
        fallback = re.search(r"//[^/]+/([^/?#]+)", profile_url)
        return fallback.group(1) if fallback else None
    except Exception as e:
        logger.error(f"❌ URL extraction error: {e}", exc_info=True)
        return None

def predict_username(username: str) -> Dict[str, str]:
    if not username or not load_username_model():
        return None
    vector = MODEL_CACHE["vectorizer"].transform([username])
    prediction = MODEL_CACHE["svm_model"].predict(vector)[0]
    result = "Real Account" if prediction == 0 else "Fake Account"
    return {"image_result": result, "advice": random.choice(ADVICE_MESSAGES[result])}

# ------------------------- Django Views -------------------------
def home(request: HttpRequest) -> HttpResponse:
    return render(request, 'home.html')

def check_profile(request: HttpRequest) -> HttpResponse:
    return render(request, "check_profile.html")

def profile_input(request: HttpRequest) -> HttpResponse:
    result = None
    error = None

    if request.method != "POST":
        return render(request, "profile_input.html")

    upload_dir = Path(settings.MEDIA_ROOT) / "uploaded_images"
    upload_dir.mkdir(exist_ok=True)

    input_type = request.POST.get("input_type", "username")
    logger.info(f"📥 Input type: {input_type}")

    try:
        if input_type == "username":
            username = request.POST.get("username", "").strip()
            result = predict_username(username) if username else None
            if not result:
                error = "Invalid or missing username."

        elif input_type == "profile_url":
            profile_url = request.POST.get("profile_url", "").strip()
            username = extract_username(profile_url)
            result = predict_username(username) if username else None
            if not result:
                error = "Could not process profile URL."

        elif input_type == "profile_image":
            uploaded_image = request.FILES.get("profile_image")
            if uploaded_image:
                try:
                    filename = default_storage.save(f"uploaded_images/{uploaded_image.name}", uploaded_image)
                    image_path = Path(settings.MEDIA_ROOT) / filename
                    result = predict_image(image_path)
                    if not result:
                        error = "Failed to analyze image."
                except Exception as e:
                    logger.error(f"❌ Error processing uploaded image: {e}", exc_info=True)
                    error = "An error occurred while processing the image."
            else:
                error = "No image uploaded."

        else:
            error = "Invalid input type selected."

    except Exception as e:
        logger.error(f"❌ Exception during form processing: {e}", exc_info=True)
        error = f"An error occurred: {str(e)}"

    return render(request, "profile_input.html", {"error": error, "result": result})

def whats_new(request):
    return render(request, 'whats_new.html')

def get_newsapi_news(request):
    api_key = 'a656d2ee524d4346862414c6e533a45d'
    query = '"online scam" OR "cyber fraud" OR "cybersecurity"'
    from_date = (datetime.today() - timedelta(days=15)).strftime('%Y-%m-%d')
    params = {
        'q': query,
        'from': from_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': api_key,
    }

    try:
        response = requests.get('https://newsapi.org/v2/everything', params=params)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        return JsonResponse([{
            'title': a.get('title'),
            'url': a.get('url'),
            'publishedAt': a.get('publishedAt')
        } for a in articles], safe=False)
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
