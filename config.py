import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///sign_language.db')
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/weights/gesture_model.h5')
    CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
