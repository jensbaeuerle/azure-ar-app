import cv2
import numpy as np
import dlib
import xlwt
from collections import OrderedDict

cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()

img_location = "images/picture.png"
book_coordiantes = xlwt.Workbook(encoding="utf-8")

while True:
    # start camera
    _, frame = cap.read()
    frame_copy = cv2.copyMakeBorder(frame,0,0,0,0,cv2.BORDER_REPLICATE)
    # convert to grey frame to save CPU power
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    detection = face_detector(gray_image)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    for coordinates in detection:

        landmarks = predictor(gray_image, coordinates)

        data_points = np.empty([2, 40])

        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x,y), 3, (0,0,255), -1)

    # display frame displaying current camera image
    cv2.imshow("Picture Frame", frame)

    key = cv2.waitKey(1)
    if key == 27: # ESC key to stop
        break
    elif key == ord('p'):
        # save coordinates into excel sheet
        sheet_landmarks = book_coordiantes.add_sheet("Lips Landmarks")
        sheet_landmarks.write(0, 0, "Point Number")
        sheet_landmarks.write(0, 1, "x")
        sheet_landmarks.write(0, 2, "y")
        res = []
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            res.append([x,y])
        n = 48
        for i in range(len(res)):
            sheet_landmarks.write(i+1, 0, n)
            sheet_landmarks.write(i+1, 1, res[i][0])
            sheet_landmarks.write(i+1, 2, res[i][1])
            n+=1
        book_coordiantes.save("coordinates/points.xls")

        #outside line
        for i,j in zip(range(48, 59), range(49,60)):
            if j == 59:
                cv2.line(frame_copy, (landmarks.part(59).x, landmarks.part(59).y), (landmarks.part(48).x,landmarks.part(48).y), (0,255,0), 2) 
            cv2.line(frame_copy, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(j).x,landmarks.part(j).y), (0,255,0), 2)

        #inside line
        for i,j in zip(range(60, 68), range(61,68)):
            if j == 67:
                cv2.line(frame_copy, (landmarks.part(67).x, landmarks.part(67).y), (landmarks.part(60).x,landmarks.part(60).y), (0,255,0), 2) 
            cv2.line(frame_copy, (landmarks.part(i).x, landmarks.part(i).y), (landmarks.part(j).x,landmarks.part(j).y), (0,255,0), 2)

        #48-54 & 60 - 64
        cv2.fillConvexPoly(frame_copy, np.array([res]), (255,153,153))
       # cv2.fillConvexPoly(frame_copy, np.array([res[7:len(res)+1]]), (178,34,34))
        cv2.imwrite(img_location, frame_copy)
        break

    #return img_location

