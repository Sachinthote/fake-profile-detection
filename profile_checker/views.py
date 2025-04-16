import os
import re
import logging
import random
import numpy as np
import cv2
from pathlib import Path
from django.shortcuts import render
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.conf import settings
from datetime import datetime, timedelta
import requests
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


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
    "image_model": BASE_DIR / "profile_checker" / "xgb_image_model.pkl",
}

MODEL_CACHE = {
    "svm_model": None,
    "vectorizer": None,
    "image_model": None,
}

FEATURE_EXTRACTOR = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
FEATURE_EXTRACTOR.trainable = False


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
    return load_model("image_model", "image_model")

# ------------------------- Prediction Logic -------------------------
def preprocess_image_for_model(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)  # Convert to array
    img = np.expand_dims(img, axis=0)  # Shape (1, 224, 224, 3)
    img = preprocess_input(img)        # Preprocess for MobileNetV2
    features = FEATURE_EXTRACTOR.predict(img, verbose=0)  # Shape (1, 1280)
    return features


def predict_image(image_path: str) -> dict:
    try:
        if not load_image_model():
            raise ValueError("Model loading failed.")

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("Image could not be read by OpenCV.")
        processed = preprocess_image_for_model(img)

        model = MODEL_CACHE["image_model"]
        prediction = model.predict(processed)[0]

        # FIXED: Reversed the condition - 1 is now "Real Account", 0 is "Fake Account"
        result = "Real Account" if prediction == 1 else "Fake Account"
        return {"image_result": result, "advice": random.choice(ADVICE_MESSAGES[result])}

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return None

def extract_username(profile_url: str):
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

def predict_username(username: str) -> dict:
    if not username or not load_username_model():
        return None
    vector = MODEL_CACHE["vectorizer"].transform([username])
    prediction = MODEL_CACHE["svm_model"].predict(vector)[0]
    # FIXED: Reversed the condition - 1 is now "Real Account", 0 is "Fake Account"
    
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

def whats_new(request: HttpRequest) -> HttpResponse:
    return render(request, 'whats_new.html')

def get_newsapi_news(request: HttpRequest) -> JsonResponse:
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