import cv2
import numpy as np
import dlib
import base64
import json
from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image
from io import BytesIO
import face_recognition
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from sklearn.cluster import KMeans
from .models import FaceAnalysis
from django.views.decorators.csrf import csrf_exempt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("analyzer/models/shape_predictor_68_face_landmarks.dat")
gender_model = load_model("analyzer/models/gender_classification_vgg16_finetuned_all_layers.keras")
beard_model = load_model("analyzer/models/beard_classification_vgg16_finetuned_all_layers.keras")

def index(request):
    return render(request, 'index.html')

from .models import FaceAnalysis

@csrf_exempt
def analyze_face(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data['image'].split(",")[1]
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Detect faces
            face_locations = face_recognition.face_locations(image_cv)
            if not face_locations:
                return JsonResponse({"error": "No face detected"}, status=400)

            # Analyze face
            sex = predict_sex(image_cv)
            beard = detect_beard(image_cv)
            eye_color = detect_eye_color(image_cv)
            skin_color = detect_skin_color(image_cv)

            # Save to the database
            analysis = FaceAnalysis(
                sex=sex,
                skin_color=skin_color,
                eye_color=eye_color,
                beard=beard
            )
            analysis.save()

            results = {
                "sex": sex,
                "beard": beard,
                "eye_color": eye_color,
                "skin_color": skin_color,
                "id": analysis.id 
            }

            return JsonResponse(results)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


def predict_sex(image):
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return "Unknown"
    top, right, bottom, left = face_locations[0]
    face = image[top:bottom, left:right]

    face = cv2.resize(face, (224, 224)) 
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)

    prediction = gender_model.predict(face)
    print(f"Prediction: {prediction}") 
    return "Male" if prediction[0][0] > 0.5 else "Female"


def detect_beard(image):
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return "Unknown"

    top, right, bottom, left = face_locations[0]
    face = image[top:bottom, left:right]

    face = cv2.resize(face, (224, 224))
    face = img_to_array(face) / 255.0
    face = np.expand_dims(face, axis=0)

    prediction = beard_model.predict(face)
    return "Yes" if prediction[0][0] > 0.5 else "No" 



def detect_eye_color(image):
    face_landmarks = face_recognition.face_landmarks(image)
    if not face_landmarks:
        return "Unknown"

    left_eye = face_landmarks[0]['left_eye']
    right_eye = face_landmarks[0]['right_eye']

    eye_pixels = []
    for eye in [left_eye, right_eye]:
        for (x, y) in eye:
            eye_pixels.append(image[y, x])

    eye_pixels = np.array(eye_pixels)
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(eye_pixels)

    dominant_color = kmeans.cluster_centers_[0]

    # Convert BGR to common eye color names
    if dominant_color[2] > 100 and dominant_color[1] > 50:
        return "Green"
    elif dominant_color[2] > 150:
        return "Blue"
    else:
        return "Brown"
  

def detect_skin_color(image):
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        return "Unknown"

    top, right, bottom, left = face_locations[0]
    face = image[top:bottom, left:right]

    pixels = face.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)

    dominant_color = kmeans.cluster_centers_[0]

    if dominant_color[0] > 180:
        return "Fair"
    elif dominant_color[0] > 100:
        return "Medium"
    else:
        return "Dark"




