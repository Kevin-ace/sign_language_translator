# Sign Language Translator

A real-time sign language translation system using computer vision and machine learning.

## Features

- Real-time hand gesture recognition
- Sign language to text translation
- User-friendly web interface
- Translation history
- Supports common signs and gestures

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
sign-language-translator/
├── app.py                  # Main Flask application
├── static/                 # Static files (CSS, JS)
│   ├── css/
│   └── js/
├── templates/             # HTML templates
├── models/               # ML model files
└── utils/               # Helper functions
```

## Technology Stack

- Frontend: HTML5, CSS3, JavaScript
- Backend: Python, Flask
- ML: TensorFlow, MediaPipe
- Computer Vision: OpenCV

## Contributing

This is an open-source project. Contributions are welcome!
