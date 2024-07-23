import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import time


def init_model():
    """Initialize the YOLO model."""
    model = YOLO('models/0w7981alf-small.pt')
    return model


def init_video_capture(video_path):
    """Initialize video capture and retrieve video properties."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    width, height, fps = (int(cap.get(attr)) for attr in
                          (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    return cap, width, height, fps


def add_annotations_to_frame(frame, count, mid_point_x, mid_point_y):
    annotator = Annotator(frame, 2)
    counter_pos = (mid_point_x - 10, 30, mid_point_x + 10, 50)
    annotator.text_label(counter_pos, f"Count :: {count}")

    region_points = [(mid_point_x - 1, mid_point_y - 250), (mid_point_x + 1, mid_point_y - 250),
                     (mid_point_x + 1, mid_point_y + 250), (mid_point_x - 1, mid_point_y + 250)]
    annotator.draw_region(reg_pts=region_points, color=(255, 0, 255), thickness=2)


def process_frame(model, frame, unique_track_ids, mid_point_x, mid_point_y):
    """Process each frame: track objects, count them, and annotate the frame."""
    tracks = model.track(frame, False, persist=True, show=False, verbose=False, conf=0.8)

    if tracks[0].boxes.id is not None:
        boxes = tracks[0].boxes.xyxy.cpu()
        clss = tracks[0].boxes.cls.cpu().tolist()
        track_ids = tracks[0].boxes.id.int().cpu().tolist()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            if track_id in unique_track_ids:
                continue

            if is_region_within_box(box, mid_point_x, mid_point_y):
                unique_track_ids.add(track_id)

    add_annotations_to_frame(frame, len(unique_track_ids), mid_point_x, mid_point_y)
    return frame, unique_track_ids


def is_region_within_box(box, mid_point_x, mid_point_y):
    top_left_x = box[0]
    top_left_y = box[1]
    bottom_right_x = box[2]
    bottom_right_y = box[3]
    return top_left_x <= mid_point_x < bottom_right_x and top_left_y <= mid_point_y < bottom_right_y


def main():
    model = init_model()
    cap, width, height, fps = init_video_capture("video/0W7981ALF.mp4")
    mid_point_x = width / 2
    mid_point_y = height / 2

    video_writer = cv2.VideoWriter("video/object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    unique_track_ids = set()
    itr = -1
    skip_factor = 8
    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        itr += 1
        if itr % skip_factor != 0:
            continue

        frame, unique_track_ids = process_frame(model, frame, unique_track_ids, mid_point_x, mid_point_y)
        video_writer.write(frame)
        print(f"total count: {len(unique_track_ids)}")

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Total time :", (end - start), "s")
