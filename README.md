
# Face Analysis System 📷🧑‍💻

A AI-powered Django-based face analysis application that uses pre-trained deep learning models to perform face analysis tasks. The system identifies the following features from an image captured from a user's face:

- Gender (Male/Female)
- Presence of a beard (Yes/No)
- Eye color (Brown, Blue, Green)
- Skin color (Fair, Medium, Dark)

## Features

🔍 Face Detection: Uses face_recognition to detect faces in images.
🧑‍🦳 Gender Classification: A fine-tuned VGG16 model for gender classification, leveraging AI.
🧔 Beard Detection: A deep learning model to predict the presence of a beard.
👁️ Eye Color Detection: Uses KMeans clustering (an AI technique) to detect the dominant eye color.
👩‍🦳 Skin Color Detection: Uses KMeans clustering to predict the dominant skin color.

## Installation

1. Clone the repository:
    ```bash
    git clone <https://github.com/MN10101/Face_Analysis.git>
    ```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1. Ensure you have PostgreSQL installed and set up.
2. Run migrations to set up the database:
    ```bash
    python manage.py migrate
    ```
3. Start the Django development server:
    ```bash
    python manage.py runserver
    ```

4. Open a browser and navigate to `http://127.0.0.1:8000` to use the face analysis system.

## Models

The models used in the system are as follows:

- **Gender Model**: Uses a fine-tuned VGG16 model to classify gender.
- **Beard Model**: A binary classifier to detect the presence of a beard.
- **Eye Color Model**: Determines the dominant eye color using KMeans clustering.
- **Skin Color Model**: Classifies skin tone using KMeans clustering.

## Usage

1. Open the face analysis page.
2. Allow camera access to capture your face.
3. Wait for the system to analyze your face and display the results, including gender, beard presence, eye color, and skin tone.

