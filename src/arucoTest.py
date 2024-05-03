#THIS CODE IS FOR TESTING NOT INCLUDING THE ARDUINO SERIAL COMMUNICATION IMPLEMENTATION
import numpy as np
import cv2
import arucoDetector as det

calib_data_path = "../calib_data/MultiMatrix.npz"
calib_data = np.load(calib_data_path)
print(calib_data.files)
camMatrix = calib_data["camMatrix"]
distCoef = calib_data["distCoef"]
rVector = calib_data["rVector"]
tVector = calib_data["tVector"]

markerSize = 13.7 #cm

def main():

    aruco_type = "DICT_7X7_250"

    cap = cv2.VideoCapture(0)
    while (1):
        ret, img = cap.read()
        img = cv2.resize(img, (640,480), interpolation= cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [corners, ids, rejected] = det.detectAruco(img, aruco_type)
        [center, topLeft, topRight, botRight, botLeft] = det.displayAruco(img, corners, ids, rejected, aruco_type)
        distance = det.distanceNpose_estimation(img, center[0], center[1], markerSize, 
                                                det.aruco_lib[aruco_type], camMatrix, distCoef)
        
        #Distance between camera and arucoMarker (For translation of robot)
        """ string = str(distance)
        print("distance: " + string)
        #Center position of the Aruco marker (For rotation of robot)
        print("Center position:")
        print("X:{}, Y{}".format(center[0], center[1])) """
        print(det.trackingAngle(center))
        cv2.imshow("image", img)

        #by pressing Esc key you can close the program
        if(cv2.waitKey(1)&0xFF==27):
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()