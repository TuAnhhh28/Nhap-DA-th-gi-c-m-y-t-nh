import cv2

# We wrap DeepFace defensively. This avoids fatal application crashes 
# if you decide to run the project on a lightweight machine without TensorFlow constraints.
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except ImportError:
    HAS_DEEPFACE = False

class EmotionDetector:
    """Standalone module combining structural face generation and emotional sentiment grids."""
    
    def __init__(self, min_face_size=(40, 40)):
        # Utilizing pre-compiled models packaged natively within OpenCV is significantly faster than CNN sweepers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.min_face_size = min_face_size

        if not HAS_DEEPFACE:
            print("Notice: 'deepface' is not installed. Emotion engine will return 'Neutral' mock values.")
            print("To enable actual AI bindings, execute: pip install deepface")

    def detect_and_classify(self, frame):
        """
        Scans the frame independently for geometric faces and evaluates cropped regions for dominant emotion.
        Returns a list of mappings: [{"bbox": (x,y,w,h), "emotion": "str"}, ...]
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # We rely on aggressive minNeighbors thresholding to prevent detecting stadium grass/seats as false positive faces
        faces = self.face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=self.min_face_size
        )
        
        results = []
        for (x, y, w, h) in faces:
            emotion_label = "Neutral"
            
            # Conditionally map to heavy Deep Learning algorithms ONLY if the library exists and the crop is valid
            if HAS_DEEPFACE:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    try:
                        # 'enforce_detection=False' bypasses DeepFace's internal detector since we already extracted 
                        # the isolated matrix crop computationally, saving heavy execution time.
                        analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False, silent=True)
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        emotion_label = analysis['dominant_emotion'].capitalize()
                    except Exception as e:
                        pass
                        
            results.append({
                "bbox": (x, y, w, h),
                "emotion": emotion_label
            })
            
        return results

    def draw_emotions(self, frame, emotion_data):
        """
        Renders distinct aesthetic visual bounding overlays for facial detection analytics.
        """
        annotated = frame.copy()
        
        for data in emotion_data:
            x, y, w, h = data["bbox"]
            emotion = data["emotion"]
            
            # Bright Pink overlays specifically distinguish Facial vectors from generic YOLO Player Body tracking
            color = (255, 0, 255) 
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            cv2.putText(annotated, emotion, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        return annotated
