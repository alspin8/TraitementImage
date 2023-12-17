import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from cvzone.SelfiSegmentationModule import SelfiSegmentation


def sepia(src_image):
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32) / 255
    # solid color
    sepia = np.ones(src_image.shape)
    sepia[:, :, 0] *= 153  # B
    sepia[:, :, 1] *= 204  # G
    sepia[:, :, 2] *= 255  # Rs
    # hadamard
    sepia[:, :, 0] *= normalized_gray  # B
    sepia[:, :, 1] *= normalized_gray  # G
    sepia[:, :, 2] *= normalized_gray  # R
    return np.array(sepia, np.uint8)


def one_detection(cascade, img, color=(255, 0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shapes = cascade.detectMultiScale(gray, 1.1, 4)
    img = cv2.ellipse(img, (shapes[0][0] + int(shapes[0][2] * 0.5), shapes[0][1] + int(shapes[0][3] * 0.5)),
                      (int(shapes[0][2] * 0.5), int(shapes[0][3] * 0.5)), 0, 0, 360, color, 4)


def eyes_detection(cascade, img, color=(255, 0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shapes = cascade.detectMultiScale(gray, 1.1, 4)
    return shapes


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

    # print(i_img.shape)

    npa = i_alpha / 255

    p_x_npa = i_img * npa

    crop_bg = img[x:(x + i_alpha.shape[0]), y:(y + i_alpha.shape[1])]

    bg_x_rnpa = crop_bg * (1 - npa)

    i = p_x_npa + bg_x_rnpa
    i = i.astype(np.uint8)

    result = img.copy().astype(np.uint8)
    result[x:(x + i_alpha.shape[0]), y:(y + i_alpha.shape[1])] = i

    return result


def get_casque(img, w):
    w = int(w * 1.5)
    ratio = img.shape[1] / img.shape[0]

    casque = cv2.resize(img, (w, w), interpolation=cv2.INTER_AREA)

    (_, casque_alpha) = cv2.threshold(cv2.cvtColor(casque, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

    casque_alpha = cv2.cvtColor(casque_alpha, cv2.COLOR_GRAY2BGR)

    return casque, casque_alpha


def get_sunglasses(img, alpha, w):
    ratio = img.shape[0] / img.shape[1]
    sunglasses = cv2.resize(img, (w, int(w * ratio)), interpolation=cv2.INTER_AREA)
    sunglasses_alpha = cv2.resize(alpha, (w, int(w * ratio)), interpolation=cv2.INTER_AREA)

    return sunglasses, sunglasses_alpha


show_casque = True
show_sunglasses = True
show_bg = True
show_snow = True
show_sepia = True


def on_click(action):
    global show_casque
    global show_sunglasses
    global show_bg
    global show_snow
    global show_sepia
    if action == "casque":
        show_casque = not show_casque
    if action == "sunglasses":
        show_sunglasses = not show_sunglasses
    if action == "bg":
        show_bg = not show_bg
    if action == "snow":
        show_snow = not show_snow
    if action == "sepia":
        show_sepia = not show_sepia


def main():
    casque = cv2.imread("./resource/casque4.png")
    background = cv2.imread("./resource/background.jpg")
    sunglasses = cv2.imread("./resource/sunglasses.png")
    alpha_sunglasses = cv2.imread("./resource/alpha.png")

    snow = cv2.VideoCapture("./resource/snow.mp4")

    face_cascade = cv2.CascadeClassifier("./resource/haarcascades/haarcascade_frontalface_alt.xml")
    eyes_cascade = cv2.CascadeClassifier("./resource/haarcascades/haarcascade_eye_tree_eyeglasses.xml")

    seg = SelfiSegmentation()

    cap = cv2.VideoCapture(0)

    global show_casque
    global show_sunglasses
    global show_bg
    global show_snow
    global show_sepia

    root = Tk()
    root.geometry('1920x1080')
    sepia_button = Button(root, text="Sepia", command=lambda: on_click("sepia"))
    sepia_button.pack()
    bg_button = Button(root, text="Background", command=lambda: on_click("bg"))
    bg_button.pack()
    casque_button = Button(root, text="Casque", command=lambda: on_click("casque"))
    casque_button.pack()
    sunglasses_button = Button(root, text="Sunglasses", command=lambda: on_click("sunglasses"))
    sunglasses_button.pack()
    snow_button = Button(root, text="Snow", command=lambda: on_click("snow"))
    snow_button.pack()
    f1 = LabelFrame(root)
    f1.pack()
    L1 = Label(f1)
    L1.pack()

    while cap.isOpened():
        ret, frame = cap.read()
        ret_snow, frame_snow = snow.read()

        try:  # Necessary to loop our snow video
            frame_snow = cv2.resize(frame_snow, (frame.shape[1], frame.shape[0]))  # No need to removeBG because already without background.
        except Exception as e:
            snow = cv2.VideoCapture("./resource/snow.mp4")
            _, frame_snow = snow.read()
            frame_snow = cv2.resize(frame_snow, (frame.shape[1], frame.shape[0]))  # No need to removeBG because already without background.
            pass

        if show_bg:
            frame = seg.removeBG(frame, cv2.resize(background, (frame.shape[1], frame.shape[0])), cutThreshold=0.5)

        if show_snow:
            frame += frame_snow

        try:
            faces = get_bounds(face_cascade, frame)
            eyes = eyes_detection(eyes_cascade, frame)
            left = eyes[0]
            for [x, y, w, h] in faces:
                print(len(eyes))
                if len(eyes) >= 2:
                    for i in range(0, len(eyes)):
                        if x < eyes[i][0] < x + w and y < eyes[i][1] < y + h:
                            if eyes[i][0] < left[0]:
                                left = eyes[i]
                img, alpha = get_casque(casque, w)
                if show_sunglasses:
                    s_img, s_alpha = get_sunglasses(sunglasses, alpha_sunglasses, int(w * 0.75))
                    frame = insert(frame, s_img, s_alpha, int(y + w * 0.25), left[0] - 15)
                    # frame = insert(frame, s_img, s_alpha, left[1] - 15, left[0] - 35)
                if show_casque:
                    frame = insert(frame, img, alpha, int(y - w * 0.25), int(x - h * 0.25))
        except Exception as e:
            print(e)
            pass

        if show_sepia:
            frame = sepia(frame)

        frame = cv2.flip(frame, 1)
        frame2 = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        L1['image'] = frame2
        root.update()
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    snow.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
