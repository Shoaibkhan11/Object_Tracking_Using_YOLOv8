import ultralytics

from ultralytics import YOLO
import cv2


def track(video_path):
    # load yolov8 model
    model = YOLO('yolov8n.pt')

    # load video

    cap = cv2.VideoCapture(video_path)

    ret = True
    # read frames
    while ret:
        ret, frame = cap.read()

        if ret:

            # detect objects
            # track objects
            results = model.track(frame, persist=True)

            frame_ = results[0].plot()

            # visualize
            cv2.imshow('frame', frame_)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()

track('temp.mp4')