import cv2
import os
from typing import Generator, Any

class VideoHandler:
    def __init__(self, input_path: str, output_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
            
        self.cap = cv2.VideoCapture(input_path)
        self.output_path = output_path
        self.writer = None
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {input_path}")
            
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure output directory exists before opening writer
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
    def read_frames(self) -> Generator[Any, None, None]:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
            
    def write_frame(self, frame):
        if self.writer is None:
            # Initialize writer lazily upon receiving first frame
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
            
        self.writer.write(frame)
        
    def release(self):
        self.cap.release()
        if self.writer:
            self.writer.release()
