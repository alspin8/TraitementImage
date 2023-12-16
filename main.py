import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation


def one_detection(cascade, img, color=(255, 0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shapes = cascade.detectMultiScale(gray, 1.1, 4)
    img = cv2.ellipse(img, (shapes[0][0] + int(shapes[0][2] * 0.5), shapes[0][1] + int(shapes[0][3] * 0.5)),
                      (int(shapes[0][2] * 0.5), int(shapes[0][3] * 0.5)), 0, 0, 360, color, 4)

    return img


def get_bounds(cascade, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, 1.1, 4)


def insert(img, i_img, i_alpha, x, y):
    if img.dtype != float:
        img = img.astype(float)
    if i_img.dtype != float:
        i_img = i_img.astype(float)
    if i_alpha.dtype != float:
        i_alpha = i_alpha.astype(float)

    print(i_img.shape)

    npa = i_alpha / 255

    p_x_npa = i_img * npa

    crop_bg = img[x:(x + i_alpha.shape[0]), y:(y + i_alpha.shape[1])]

    bg_x_rnpa = crop_bg * (1 - npa)

    i = p_x_npa + bg_x_rnpa
    i = i.astype(np.uint8)

    result = img.copy().astype(np.uint8)
    result[x:(x + i_alpha.shape[0]), y:(y + i_alpha.shape[1])] = i

    return result


def get_image(img, w):
    w = int(w * 1.5)
    ratio = img.shape[1] / img.shape[0]

    casque = cv2.resize(img, (w, w), interpolation=cv2.INTER_AREA)

    (_, casque_alpha) = cv2.threshold(cv2.cvtColor(casque, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

    casque_alpha = cv2.cvtColor(casque_alpha, cv2.COLOR_GRAY2BGR)

    return casque, casque_alpha


def main():
    casque = cv2.imread("./resource/casque4.png")
    background = cv2.imread("./resource/background.jpg")

    face_cascade = cv2.CascadeClassifier("./resource/haarcascades/haarcascade_frontalface_alt.xml")

    seg = SelfiSegmentation()

    cv2.namedWindow("dd")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        frame = seg.removeBG(frame, cv2.resize(background, (frame.shape[1], frame.shape[0])), cutThreshold=0.5)

        try:
            faces = get_bounds(face_cascade, frame)
            for [x, y, w, h] in faces:
                img, alpha = get_image(casque, w)
                frame = insert(frame, img, alpha, int(y - w * 0.25), int(x - h * 0.25))
        except Exception as e:
            print(e)
            pass

        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
