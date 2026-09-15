import os
import argparse
import urllib.request

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
DEFAULT_MODEL_PATH = "blaze_face_short_range.tflite"


def ensure_model(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """Download the face detector model if it isn't already present locally."""
    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}', downloading...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the face detector model automatically ({exc}). "
                f"Download it manually from {MODEL_URL} and place it at '{model_path}', "
                "or pass --model with the correct path."
            ) from exc
    return model_path


def build_detector(model_path: str) -> FaceDetector:
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    return FaceDetector.create_from_options(options)


def blur_faces(frame, detector: FaceDetector):
    """Detect faces in a BGR frame and blur each detected face region in place."""
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    for detection in result.detections:
        bbox = detection.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        box_w, box_h = bbox.width, bbox.height

        # Clamp to frame bounds: the model can return boxes that extend
        # past the image edges, which would otherwise produce a
        # negative/empty slice and crash cv2.blur.
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x1 + box_w)
        y2 = min(h, y1 + box_h)

        if x2 <= x1 or y2 <= y1:
            continue

        frame[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (30, 30))

    return frame


def main():
    parser = argparse.ArgumentParser(description="Blur faces in an image, video, or webcam feed.")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], default="webcam")
    parser.add_argument("--filePath", default=None, help="Required for --mode image/video")
    parser.add_argument("--camIndex", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the .task model file")
    args = parser.parse_args()

    if args.mode in ("image", "video") and not args.filePath:
        parser.error(f"--filePath is required when --mode is '{args.mode}'")

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    model_path = ensure_model(args.model)

    with build_detector(model_path) as detector:
        if args.mode == "image":
            img = cv2.imread(args.filePath)
            if img is None:
                raise FileNotFoundError(f"Could not read image at '{args.filePath}'")
            img = blur_faces(img, detector)
            cv2.imwrite(os.path.join(output_dir, "output.png"), img)

        elif args.mode == "video":
            cap = cv2.VideoCapture(args.filePath)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video at '{args.filePath}'")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Video has no readable frames")

            output_video = cv2.VideoWriter(
                os.path.join(output_dir, "output.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame.shape[1], frame.shape[0]),
            )

            try:
                while ret:
                    frame = blur_faces(frame, detector)
                    output_video.write(frame)
                    ret, frame = cap.read()
            finally:
                cap.release()
                output_video.release()

        elif args.mode == "webcam":
            cap = cv2.VideoCapture(args.camIndex)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open webcam at index {args.camIndex}")

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = blur_faces(frame, detector)
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()