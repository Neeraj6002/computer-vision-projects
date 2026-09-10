import cv2

from util import get_limits


# Start webcam
cap = cv2.VideoCapture(0)

# Orange in BGR
orange = [0, 165, 255]


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read from webcam")
        break

    # Convert BGR to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV limits
    lowerLimit, upperLimit = get_limits(orange)

    # Create mask
    mask = cv2.inRange(
        hsvImage,
        lowerLimit,
        upperLimit
    )

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Check each detected object
    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore small objects/noise
        if area > 1000:

            x, y, w, h = cv2.boundingRect(contour)

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                5
            )

    # Display
    cv2.imshow("Orange Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()