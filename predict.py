import cv2 as cv
import numpy as np
from time import time
from plate import score_frame, plot_boxes
import os


def main_loop(video_path='', output_folder=''):

    # saving the predicted video
    video = cv.VideoCapture(
        'demo/VID-20230502-WA0010.mp4')
    writer = cv.VideoWriter(filename="".join(['results', os.sep, "VID-20230502-WA0010.MP4"]),
                            apiPreference=cv.CAP_FFMPEG,
                            params=None,
                            fourcc=cv.VideoWriter_fourcc(*"mp4v"),
                            fps=int(video.get(cv.CAP_PROP_FPS)),
                            frameSize=(int(video.get(cv.CAP_PROP_FRAME_WIDTH)),
                                       int(video.get(cv.CAP_PROP_FRAME_HEIGHT))))
    # initialization
    frame_counter = 0
    ocr_interval = 4  # 4 frame per ocr
    ocr_access = False
    ocr_history = {}

    # parameters
    score_plate = 0.75  # default >0.8
    score_ocr = 0.90  # default >0.98
    min_x1 = 10  # crop range from left side
    min_x2 = 10  # crop range from right side
    min_y1 = 10  # crop range from bottom side
    min_y2 = 10  # crop range from top side
    quality = (20, 50)  # default for OCR: (32, 100)
    hp = {'score_plate': score_plate,
          'score_ocr': score_ocr,
          'min_x1': min_x1,
          'min_x2': min_x2,
          'min_y1': min_y1,
          'min_y2': min_y2,
          'quality': quality}
    text = ''

    number = ''
    while video.isOpened():
        camera, frame = video.read()

        if not camera:
            print("[WARNING] cannot read the video any further")
            break
        start_time = time()

        # plate detection
        if camera:
            if frame_counter % ocr_interval == 0:
                results1 = score_frame([frame])

            # plotting plate and detection OCR
            if len(results1[0]):
                frame, text = plot_boxes(
                    results1, frame, hp)

            # saving texts into dict
            if text:
                ocr_history[text] = ocr_history.get(text, 0) + 1

            # getting most frequent text
            if ocr_history:
                number = max(ocr_history, key=ocr_history.get)
                print(max(ocr_history, key=ocr_history.get))

            # calculating fps
            end_time = time()
            fps = 1 / np.round(end_time - start_time, 3)
            if fps == float('inf'):
                fps = 30
            # visualizing fps
            cv.putText(frame, number, (0, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv.LINE_AA)

        cv.imshow("video", frame)

        # visualizing fps
        # cv.putText(frame2, f"FPS: {np.round(fps)}", (0, 60),
        #           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        # cv.putText(frame, f"W: {get_weight()}", (0, 30),
        #           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        writer.write(frame)
        # frame interval for OCR
        frame_counter += 1
        number = ''

        # exit on a keypress: q
        if cv.waitKey(1) == 113:
            break

    video.release()
    writer.release()
    cv.destroyAllWindows()


main_loop()
