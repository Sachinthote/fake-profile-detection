import os
import re
import logging
import random
import numpy as np
import joblib
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

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent

# Model paths
MODEL_PATHS = {
    "username_model": BASE_DIR / "profile_checker" / "svm_model.pkl",
    "vectorizer": BASE_DIR / "profile_checker" / "vectorizer.pkl",
    "image_model": BASE_DIR / "profile_checker" / "svm_image_model.pkl",
    "feature_extractor": BASE_DIR / "profile_checker" / "feature_extractor.pkl",
}

# Model cache
MODEL_CACHE = {
    "svm_model": None,
    "vectorizer": None,
    "image_svm_model": None,
    "feature_extractor": None,
}

# URL patterns
URL_PATTERNS = {
    "twitter": r"(?:twitter|x)\.com/([^/?#]+)",
    "instagram": r"instagram\.com/([^/?#]+)",
    "facebook": r"facebook\.com/([^/?#]+)",
    "tiktok": r"tiktok\.com/@?([^/?#]+)",
    "linkedin": r"linkedin\.com/in/([^/?#]+)",
}

# Advice messages
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

def load_model(model_key: str, path_key: str) -> bool:
    if MODEL_CACHE[model_key] is None:
        try:
            MODEL_CACHE[model_key] = joblib.load(MODEL_PATHS[path_key])
            logger.info(f"✅ Successfully loaded {model_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading {model_key}: {e}", exc_info=True)
            return False
    return True

def load_username_model() -> bool:
    return (load_model("svm_model", "username_model") and
            load_model("vectorizer", "vectorizer"))

def load_image_model() -> bool:
    if not load_model("image_svm_model", "image_model"):
        return False
    if MODEL_CACHE["feature_extractor"] is None:
        try:
            MODEL_CACHE["feature_extractor"] = joblib.load(MODEL_PATHS["feature_extractor"])
            logger.info("✅ Successfully loaded feature extractor model")
        except Exception as e:
            logger.error(f"❌ Error loading feature extractor: {e}", exc_info=True)
            return False
    return True

def extract_username(profile_url: str) -> Optional[str]:
    try:
        if not profile_url:
            return None
        if not profile_url.startswith(('http://', 'https://')):
            profile_url = 'https://' + profile_url
        profile_url = profile_url.lower()
        for platform, pattern in URL_PATTERNS.items():
            match = re.search(pattern, profile_url)
            if match:
                return match.group(1)
        simple_username = re.search(r"//[^/]+/([^/?#]+)", profile_url)
        return simple_username.group(1) if simple_username else None
    except Exception as e:
        logger.error(f"❌ Error extracting username: {e}", exc_info=True)
        return None

def extract_features(image_path: str) -> np.ndarray:
    try:
        if not load_image_model():
            raise ValueError("Failed to load image model or feature extractor.")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Uploaded image not found: {image_path}")

        # Preprocess manually
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg16.preprocess_input(img)

        features = MODEL_CACHE["feature_extractor"].predict(img)
        return features.flatten().reshape(1, -1)
    except Exception as e:
        logger.error(f"❌ Error extracting image features: {e}", exc_info=True)
        raise ValueError(f"Feature extraction failed: {str(e)}")

def get_random_advice(result_type: str) -> str:
    advice_list = ADVICE_MESSAGES.get(result_type, ADVICE_MESSAGES["default"])
    return random.choice(advice_list)

def predict_username(username: str) -> Dict[str, str]:
    if not username or not load_username_model():
        return None
    vectorized = MODEL_CACHE["vectorizer"].transform([username])
    prediction = MODEL_CACHE["svm_model"].predict(vectorized)[0]
    result = "Real Account" if prediction == 0 else "Fake Account"
    return {
        "image_result": result,
        "advice": get_random_advice(result)
    }

def predict_image(image_path: str) -> Dict[str, str]:
    try:
        features = extract_features(image_path)
        prediction = MODEL_CACHE["image_svm_model"].predict(features)[0]
        result = "Real Account" if prediction == 0 else "Fake Account"
        return {
            "image_result": result,
            "advice": get_random_advice(result)
        }
    except Exception as e:
        logger.error(f"❌ Error during image prediction: {e}", exc_info=True)
        return None

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
    logger.info(f"Processing input type: {input_type}")

    try:
        if input_type == "username":
            username = request.POST.get("username", "").strip()
            if username:
                result = predict_username(username)
                if not result:
                    error = "Failed to process username."
            else:
                error = "Please enter a valid username."

        elif input_type == "profile_url":
            profile_url = request.POST.get("profile_url", "").strip()
            username = extract_username(profile_url)
            if username:
                result = predict_username(username)
                if not result:
                    error = "Failed to process profile URL."
            else:
                error = "Invalid profile URL or could not extract username."

        elif input_type == "profile_image":
            uploaded_image = request.FILES.get("profile_image")
            if uploaded_image:
                filename = default_storage.save(
                    f"uploaded_images/{uploaded_image.name}", uploaded_image
                )
                image_path = Path(settings.MEDIA_ROOT) / filename
                result = predict_image(image_path)
                if not result:
                    error = "Failed to analyze the image."
            else:
                error = "Please upload a valid image."

        else:
            error = "Invalid input type."

    except Exception as e:
        logger.error(f"❌ Unexpected error in profile_input: {e}", exc_info=True)
        error = f"An unexpected error occurred: {str(e)}"

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

    url = 'https://newsapi.org/v2/everything'
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        news_list = [{
            'title': a.get('title'),
            'url': a.get('url'),
            'publishedAt': a.get('publishedAt')
        } for a in data.get('articles', [])]
        return JsonResponse(news_list, safe=False)
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)
