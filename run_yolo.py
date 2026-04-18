import cv2
from ultralytics import YOLO

def main():
    # Load the pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")
    
    # Open the local video file
    video_path = "input.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}.")
        return

    # Get video properties for the output writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Handle invalid FPS values (e.g. 0.0 or NaN)
    if fps == 0 or fps != fps:
        fps = 30.0
    fps = int(fps)
    
    # Define the codec and create a VideoWriter object
    # Changing output from mp4v/mp4 to XVID/avi for better player/IDE compatibility
    output_path = "output.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # COCO class IDs: 0 is 'person', 32 is 'sports ball'
    target_classes = [0, 32]
    
    print("Starting detection. This may take a while...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Ensure the current frame size perfectly matches the VideoWriter's target size
        # This handles dynamic frame sizing issues if the stream changes midway
        frame = cv2.resize(frame, (width, height))
            
        # Run YOLOv8 detection, filtering for specific classes
        results = model(frame, classes=target_classes, verbose=False)
        
        # Plot the bounding boxes and labels on the frame
        annotated_frame = results[0].plot()
        
        # Save the annotated frame to the output video
        out.write(annotated_frame)
        
    # Release resources
    cap.release()
    out.release()
    
    print(f"Processing complete. Annotated video saved as {output_path}")

if __name__ == "__main__":
    main()
