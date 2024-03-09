import numpy as np
import cv2
import arucoDetector as det

def main():

    aruco_type = "DICT_7X7_250"

    cap = cv2.VideoCapture(0)
    while (1):
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [corners, ids, rejected] = det.detectAruco(img, aruco_type)
        [center, topLeft, topRight, botRight, botLeft] = det.displayAruco(img, corners, ids, rejected, aruco_type)
        cv2.imshow("image", img)

        #by pressing Esc key you can close the program
        if(cv2.waitKey(1)&0xFF==27):
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()