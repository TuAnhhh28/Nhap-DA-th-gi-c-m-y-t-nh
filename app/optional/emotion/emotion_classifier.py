# Wrapper attempting to import DeepFace, failing gracefully to avoid blocking the user 
# if they haven't run `pip install deepface`.
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

class EmotionClassifier:
    """Classifies isolated facial crops utilizing pre-trained VGG-Face or simplistic models."""
    
    def __init__(self):
        if not HAS_DEEPFACE:
            print("Warning: 'deepface' library not detected on system. EmotionClassifier will mock a 'Neutral' state. Run 'pip install deepface' to activate actual AI bindings.")
            
    def analyze(self, frame, face_boxes):
        """
        Receives frame dimensions and crops exact facial pixel data, 
        running it linearly through the classification model.
        """
        results = []
        for (x, y, w, h) in face_boxes:
            # Slicing the numpy array isolates only the facial pixel data for extreme speed processing
            face_crop = frame[y:y+h, x:x+w]
            emotion_label = "Neutral" # Fallback standard
            
            # Prevent crashes by checking dimensions
            if HAS_DEEPFACE and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                try:
                    # 'enforce_detection=False' bypasses DeepFace's internal detector since we already cropped explicitly
                    analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                    
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    emotion_label = analysis['dominant_emotion'].capitalize()
                except Exception as e:
                    pass
            
            results.append({
                "face_bbox": [x, y, x+w, y+h], # Converted to x1, y1, x2, y2 format immediately for draw utilities
                "emotion": emotion_label
            })
            
        return results
