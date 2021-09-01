import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):
        """
        Draw landmarks and boxes for each face detected
        """
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                box = box.astype('int')
                # Draw rectangle on frame
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)

                # Show probability
                cv2.putText(frame, str(
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                # cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except:
            pass

        return frame

    def run(self, img):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """

        # detect face box, probability and landmarks
        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        # draw on frame
        self._draw(img, boxes, probs, landmarks)

        while True:
            # Show the frame
            cv2.imshow('Face Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        
        
# Run the app
mtcnn = MTCNN()
# img = cv2.imread('/home/pokealimit/extracted_image/NIE_CCW_img/NIE_CCW_0037.jpg')
# img = cv2.imread('/mnt/c/Users/Da Wei/Downloads/TTHS1VR.jpg')
img = cv2.imread('/home/pokealimit/rosbags/medicineBuilding_img/frame0107.jpg')

# while True:
#     cv2.imshow('image', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


fcd = FaceDetector(mtcnn)
fcd.run(img)