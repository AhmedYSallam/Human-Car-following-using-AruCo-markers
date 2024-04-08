import numpy as np
import cv2
import arucoDetector as det
import serial,time

serialcomm = serial.Serial('COM5', 115200)
serialcomm.timeout = 3
time.sleep(1)

calib_data_path = "calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
print(calib_data.files)
camMatrix = calib_data["camMatrix"]
distCoef = calib_data["distCoef"]
rVector = calib_data["rVector"]
tVector = calib_data["tVector"]

markerSize = 13.7 #cm

def main():

    aruco_type = "DICT_7X7_250"

    cap = cv2.VideoCapture(1)
    while (1):
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [corners, ids, rejected] = det.detectAruco(img, aruco_type)
        [center, topLeft, topRight, botRight, botLeft] = det.displayAruco(img, corners, ids, rejected, aruco_type)
        distance = det.distanceNpose_estimation(img, center[0], center[1], markerSize, 
                                                det.aruco_lib[aruco_type], camMatrix, distCoef)
        dis = str((distance))
        print("from python: " + dis)
        serialcomm.write((bytes(dis, 'utf-8')))
        print(serialcomm.readline().decode('utf-8').rstrip())
        cv2.imshow("image", img)

        #by pressing Esc key you can close the program
        if(cv2.waitKey(1)&0xFF==27):
            break
    serialcomm.close()
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()