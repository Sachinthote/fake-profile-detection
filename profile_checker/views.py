import os
import re
import logging
import functools
import random
import numpy as np
from django.http import JsonResponse
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Union, Tuple

import joblib
import cv2
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.http import JsonResponse, HttpRequest, HttpResponse
from django.conf import settings
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer

# Lazy imports for TensorFlow to improve startup time
# Will only be imported when needed
tf_imported = False
vgg16_module = None
preprocessing_module = None

# Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths for models
BASE_DIR = Path(__file__).resolve().parent.parent

# File paths
MODEL_PATHS = {
    "username_model": BASE_DIR / "profile_checker" / "svm_model.pkl",
    "vectorizer": BASE_DIR / "profile_checker" / "vectorizer.pkl",
    "image_model": BASE_DIR / "profile_checker" / "svm_image_model.pkl",
}

# Cache for loaded models
MODEL_CACHE = {
    "svm_model": None,
    "vectorizer": None,
    "image_svm_model": None,
    "vgg_model": None
}

# Social media URL patterns
URL_PATTERNS = {
    "twitter": r"(?:twitter|x)\.com/([^/?#]+)",
    "instagram": r"instagram\.com/([^/?#]+)",
    "facebook": r"facebook\.com/([^/?#]+)",
    "tiktok": r"tiktok\.com/@?([^/?#]+)",
    "linkedin": r"linkedin\.com/in/([^/?#]+)",
}

# Multiple advice messages for different scenarios
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


def lazy_load_tf():
    """Lazily import TensorFlow modules only when needed"""
    global tf_imported, vgg16_module, preprocessing_module
    
    if not tf_imported:
        try:
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
            from tensorflow.keras.preprocessing.image import load_img, img_to_array
            
            vgg16_module = {"VGG16": VGG16, "preprocess_input": preprocess_input}
            preprocessing_module = {"load_img": load_img, "img_to_array": img_to_array}
            tf_imported = True
            
            return True
        except ImportError as e:
            logger.error(f"❌ Failed to import TensorFlow modules: {e}", exc_info=True)
            return False
    return True


def load_model(model_key: str, path_key: str) -> bool:
    """Generic model loader with caching"""
    if MODEL_CACHE[model_key] is None:
        try:
            MODEL_CACHE[model_key] = joblib.load(MODEL_PATHS[path_key])
            logger.info(f"✅ Successfully loaded {model_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Error loading {model_key}: {e}", exc_info=True)
            return False
    return True


# 🆕 Render the "What's New" page
def whats_new(request):
    return render(request, 'whats_new.html')

# 🆕 API endpoint to return cybersecurity news



def get_newsapi_news(request):
    api_key = 'a656d2ee524d4346862414c6e533a45d'

    # Build query
    query = '"online scam" OR "cyber fraud" OR "cybersecurity"'
    from_date = (datetime.today() - timedelta(days=15)).strftime('%Y-%m-%d')

    # Prepare parameters safely
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

        news_list = []
        for article in data.get('articles', []):
            news_item = {
                'title': article.get('title'),
                'url': article.get('url'),
                'publishedAt': article.get('publishedAt')
            }
            news_list.append(news_item)

        return JsonResponse(news_list, safe=False)
    except requests.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)



def load_username_model() -> bool:
    """Load the username-based SVM model and vectorizer."""
    return (load_model("svm_model", "username_model") and 
            load_model("vectorizer", "vectorizer"))


def load_image_model() -> bool:
    """Load the image-based SVM model and VGG16 feature extractor."""
    if not load_model("image_svm_model", "image_model"):
        return False
    
    if MODEL_CACHE["vgg_model"] is None:
        if not lazy_load_tf():
            return False
            
        try:
            MODEL_CACHE["vgg_model"] = vgg16_module["VGG16"](
                weights="imagenet", 
                include_top=False, 
                input_shape=(224, 224, 3)
            )
            logger.info("✅ Successfully loaded VGG16 model")
        except Exception as e:
            logger.error(f"❌ Error loading VGG16 model: {e}", exc_info=True)
            return False
    
    return True


def extract_username(profile_url: str) -> Optional[str]:
    """Extract the username from a given social media profile URL."""
    try:
        if not profile_url:
            logger.warning("Empty profile URL provided")
            return None

        # Normalize URL
        if not profile_url.startswith(('http://', 'https://')):
            profile_url = 'https://' + profile_url

        profile_url = profile_url.lower()
        logger.info(f"Processing URL: {profile_url}")

        # Try matching with known patterns
        for platform, pattern in URL_PATTERNS.items():
            match = re.search(pattern, profile_url)
            if match:
                username = match.group(1)
                logger.info(f"Extracted {platform} username: {username}")
                return username

        # Fallback to simple extraction
        simple_username = re.search(r"//[^/]+/([^/?#]+)", profile_url)
        if simple_username:
            username = simple_username.group(1)
            logger.info(f"Extracted username using simple pattern: {username}")
            return username

        logger.warning(f"No username found in URL: {profile_url}")
        return None
    except Exception as e:
        logger.error(f"❌ Error extracting username: {e}", exc_info=True)
        return None


def extract_features(image_path: str) -> np.ndarray:
    """Extract features from an image using VGG16."""
    try:
        if not load_image_model():
            raise ValueError("Failed to load VGG16 model")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Uploaded image file not found: {image_path}")

        # Load and preprocess the image
        load_img = preprocessing_module["load_img"]
        img_to_array = preprocessing_module["img_to_array"]
        preprocess_input = vgg16_module["preprocess_input"]
        
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features
        features = MODEL_CACHE["vgg_model"].predict(img_array)
        return features.flatten().reshape(1, -1)
    except Exception as e:
        logger.error(f"❌ Error extracting image features: {e}", exc_info=True)
        raise ValueError(f"Feature extraction failed: {str(e)}")


def get_random_advice(result_type: str) -> str:
    """Get a random piece of advice based on the result type"""
    advice_list = ADVICE_MESSAGES.get(result_type, ADVICE_MESSAGES["default"])
    return random.choice(advice_list)


def predict_username(username: str) -> Dict[str, str]:
    """Make prediction based on username"""
    if not username or not load_username_model():
        return None
        
    vectorized_username = MODEL_CACHE["vectorizer"].transform([username])
    prediction = MODEL_CACHE["svm_model"].predict(vectorized_username)[0]
    result = "Real Account" if prediction == 0 else "Fake Account"
    
    # Get random advice based on result type
    advice = get_random_advice(result)
    
    return {
        "image_result": result,
        "advice": advice
    }


def predict_image(image_path: str) -> Dict[str, str]:
    """Make prediction based on image features"""
    try:
        features = extract_features(image_path)
        prediction = MODEL_CACHE["image_svm_model"].predict(features)[0]
        result = "Real Account" if prediction == 0 else "Fake Account"
        
        # Get random advice based on result type
        advice = get_random_advice(result)
        
        return {
            "image_result": result,
            "advice": advice
        }
    except Exception as e:
        logger.error(f"❌ Error during image prediction: {e}", exc_info=True)
        return None


def home(request: HttpRequest) -> HttpResponse:
    """Home page view"""
    return render(request, 'home.html')


def check_profile(request: HttpRequest) -> HttpResponse:
    """Profile checker view"""
    return render(request, "check_profile.html")


def profile_input(request: HttpRequest) -> HttpResponse:
    """Handle profile input form submission"""
    result = None
    error = None
    
    if request.method != "POST":
        return render(request, "profile_input.html")
    
    # Ensure upload directory exists
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
                # Save the uploaded image
                filename = default_storage.save(
                    f"uploaded_images/{uploaded_image.name}", 
                    uploaded_image
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