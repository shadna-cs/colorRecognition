import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    colors = {
        "Red": [
            (np.array([0,120,70]), np.array([10,255,255])),
            (np.array([170,120,70]), np.array([180,255,255]))
        ],
        "Green": [
            (np.array([36,50,70]), np.array([89,255,255]))
        ],
        "Blue": [
            (np.array([90,50,70]), np.array([128,255,255]))
        ],
        "Yellow": [
            (np.array([20,100,100]), np.array([35,255,255]))
        ]
    }

    for color_name, ranges in colors.items():

        mask = None

        for lower, upper in ranges:
            current_mask = cv2.inRange(hsv, lower, upper)

            if mask is None:
                mask = current_mask
            else:
                mask = mask + current_mask

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            if cv2.contourArea(contour) > 800:

                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)

                cv2.putText(
                    frame,
                    color_name,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,255),
                    2
                )

    cv2.imshow("Color Recognition Project", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()