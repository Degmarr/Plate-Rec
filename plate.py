import torch
import numpy as np
import cv2
import math
from ocr.eval import demo


def load_model():
    """Loading Plate detection module.

    Returns:
      model
    """
    model = torch.hub.load(
        'yolov7', 'custom', path_or_model='best.pt', source='local')
    return model


def score_frame(frame):
    """Detecting plate from frame.

    Args:
      frame: 

    Returns:
      labels: The labels of plate.
      cord: The coordinates and score of plate.
    """
    model.to(device)
    results = model(frame)
    print(results)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    print(labels, cord)
    return labels, cord


def crop_test(x_shape, y_shape, x1, y1, x2, y2, hp):
    """Checking plate box position in frame. 
    Plate box must not be closer to end of frame sides.

    Args:
      x_shape: frame or plate.
      y_shape: default.
      x1: from left side of frame.
      y1: from bottom side of frame.
      x2: from right side of frame.
      y2: from top side of frame.
      hp: hyperparameters.

    Returns:
      True/False
    """
    return not (x1 > hp['min_x1'] and y1 > hp['min_y1'] and x2 < x_shape-hp['min_x2'] and y2 < x_shape-hp['min_y2'])


def quality_test(x1, y1, x2, y2, hp):
    """Checking plate box's quality. 

    Args:
      x1: from left side of frame.
      y1: from bottom side of frame.
      x2: from right side of frame.
      y2: from top side of frame.
      hp: hyperparameters.

    Returns:
      True/False
    """
    return y2-y1 <= hp['quality'][0] and x2-x1 <= hp['quality'][1]


def plot_boxes(results, frame, hp):
    """Plotting plate boxes into frame. 
    Evaluating OCR module. 

    Args:
      results: labels, coord from Plate Detection.
      frame: frame from the video.
      hp: hyperparameters.
      ocr_access: access for ultilizing OCR module.

    Returns:
      frame: frame with plotted plate box.
      main_text: detected OCR module's text.
    """
    labels, cord = results
    plate = cord[0]
    main_text = ""
    if plate[4] > hp['score_plate']:
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        bgr = (0, 255, 0)
        x1, y1, x2, y2 = int(plate[0]*x_shape), int(plate[1] *
                                                    y_shape), int(plate[2]*x_shape), int(plate[3]*y_shape)
        if quality_test(x1, y1, x2, y2, hp) \
                or crop_test(x_shape, y_shape, x1, y1, x2, y2, hp):
            return frame, main_text

        plate_box = frame[y1:y2, x1:x2]

        text, p = demo(plate_box)

        main_text = "".join(
            [i.upper() if not i.isdigit() else i for i in text])

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
        cv2.putText(frame, f'plate: {main_text}', (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2)
    return frame, main_text


model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

frame1 = cv2.imread('photo_2023-07-05 20.23.46.jpeg')
results1 = score_frame([frame1])
score_plate = 0.70  # default >0.8
score_ocr = 0.98  # default >0.98
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
text1, text2 = '', ''

frame1, text1 = plot_boxes(results1, frame1, hp)

cv2.putText(frame1, f"Number: {text1}", (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
cv2.imwrite('res.png', frame1)
