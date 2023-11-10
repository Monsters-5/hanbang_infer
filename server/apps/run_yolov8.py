from __future__ import annotations
import os
import statistics

import numpy as np
import supervision as sv

def start_infer(model, id):
    ## settings
    HOME = os.getcwd()
    # 수정하기
    # input 비디오 이름 설정
    SOURCE_VIDEO_PATH = f"{HOME}/server/data/{id}.mp4"
    # output 비디오 이름 설정
    TARGET_VIDEO_PATH = f"{HOME}/server/data/{id}_result.mp4"
    # initiate polygon zone
    polygon = np.array([
        [100, 700],
        [100, 100],
        [1100, 100],
        [1100, 700]
    ])
    
    ## process video
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)

    # initiate annotators
    box_annotator = sv.BoxAnnotator(thickness=3, text_thickness=2, text_scale=1)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.white(), thickness=3, text_thickness=3, text_scale=1)

    # initiate reponse data
    yolo_res = []
    
    # collback for process_video
    def process_frame(frame: np.ndarray, i) -> np.ndarray:
        # detect
        results = model(frame, imgsz=1280)[0]
        detections = sv.Detections.from_yolov8(results)
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.55)]
        zone.trigger(detections=detections)

        # annotate
        box_annotator = sv.BoxAnnotator(thickness=3, text_thickness=2, text_scale=1)
        labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = zone_annotator.annotate(scene=frame)
        
        # progress notation
        print('Frame:', (i+1))
        print('Tent_num:', zone.current_count) # 개수가 저장되어 있는 곳
        
        # save results
        yolo_res.append(int(zone.current_count))
        return frame
    
    process_video(source_path=SOURCE_VIDEO_PATH, target_path=TARGET_VIDEO_PATH, callback=process_frame)
    
    # 1초동안 측정된 개수의 중앙값과 timestamp return
    yolo_res = statistics.median(yolo_res)
    return yolo_res

### from supervision
from typing import Callable, Generator
import cv2

def get_video_frames_generator(source_path: str) -> Generator[np.ndarray, None, None]:
    """
    Get a generator that yields the frames of the video.

    Args:
        source_path (str): The path of the video file.

    Returns:
        (Generator[np.ndarray, None, None]): A generator that yields the frames of the video.

    Examples:
        ```python
        >>> from supervision import get_video_frames_generator

        >>> for frame in get_video_frames_generator(source_path='source_video.mp4'):
        ...     ...
        ```
    """
    video = cv2.VideoCapture(source_path)
    if not video.isOpened():
        raise Exception(f"Could not open video at {source_path}")
    # 가장 마지막 동영상의 프레임 정보
    last_frame_info = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # 가장 마지막 90프레임(test동영상 10230프레임 이후부터 읽어지지 않는 오류)
    for i in range(last_frame_info - (90+80), last_frame_info-80):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if success:
            yield frame
        else:
            print(f"Failed to read frame {i} from the video")
            break
    video.release()


def process_video(
    source_path: str,
    target_path: str,
    callback: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """
    Process a video file by applying a callback function on each frame and saving the result to a target video file.

    Args:
        source_path (str): The path to the source video file.
        target_path (str): The path to the target video file.
        callback (Callable[[np.ndarray, int], np.ndarray]): A function that takes in a numpy ndarray representation of a video frame and an int index of the frame and returns a processed numpy ndarray representation of the frame.

    Examples:
        ```python
        >>> from supervision import process_video

        >>> def process_frame(scene: np.ndarray) -> np.ndarray:
        ...     ...

        >>> process_video(
        ...     source_path='source_video.mp4',
        ...     target_path='target_video.mp4',
        ...     callback=process_frame
        ... )
        ```
    """
    source_video_info = sv.VideoInfo.from_video_path(video_path=source_path)
    with sv.VideoSink(target_path=target_path, video_info=source_video_info) as sink:
        for index, frame in enumerate(
            get_video_frames_generator(source_path=source_path)
        ):
            result_frame = callback(frame, index)
            sink.write_frame(frame=result_frame)


if __name__ == '__main__':
    from ultralytics import YOLO
    HOME = os.getcwd()
    MODEL = f"{HOME}/server/weights/best.pt"
    model = YOLO(MODEL)
    res = start_infer(model)
    print(res)