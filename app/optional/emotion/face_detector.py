import cv2

class FaceDetector:
    """Lightweight Haar Cascade face detector fallback native to OpenCV."""
    
    def __init__(self, min_size: tuple = (40, 40)):
        # Utilizing pre-compiled model packaged natively within OpenCV avoids heavy library dependencies
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.min_size = min_size
        
        if self.face_cascade.empty():
            print("Warning: HaarCascade XML not found. Ensure OpenCV is installed properly.")
        
    def detect(self, frame):
        """
        Detects faces mathematically in a grayscale clone of the frame.
        Filter out small hits mathematically to eliminate grass artifacts/false positives.
        Returns a list of OpenCV geometric face bounding boxes (x, y, w, h).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # scaleFactor and minNeighbors act as aggressive confidence thresholds
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=self.min_size
        )
        
        return faces
