import numpy as np
import time
import cv2
aruco_lib = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

aruco_type = "DICT_7X7_250"

def detectAruco(img):
    arucoDict = cv2.aruco.Dictionary_get(aruco_lib[aruco_type])
    arucoParam = cv2.aruco.DetectorParameters_create()
    corners, ids, rej = cv2.aruco.detectMarkers(img, 
                                                  arucoDict, 
                                                  parameters=arucoParam)
    #print(corners)
    #print(ids)
    #print(rej)
    return [corners, ids, rej]

def displayAruco(img, corners, ids, rejected):
    cx = 0
    cy = 0
    topLeft = 0
    botRight = 0
    botLeft = 0
    topRight = 0
    if(len(corners) > 0):
        ids = ids.flatten()
        for (markerCorner, ID) in zip(corners, ids):
            
            corners = markerCorner.reshape((4,2))
            #print(corners)
            [topLeft, topRight, botRight, botLeft] = corners
            
            topLeft = [int(topLeft[0]), int(topLeft[1])]
            topRight = [int(topRight[0]), int(topRight[1])]
            botRight = [int(botRight[0]), int(botRight[1])]
            botLeft = [int(botLeft[0]), int(botLeft[1])]
            
			#draw box around aruco marker
            cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(img, botRight, botLeft, (0, 255, 0), 2)
            cv2.line(img, topRight, botRight, (0, 255, 0), 2)
            cv2.line(img, botLeft, topLeft, (0, 255, 0), 2)
            
			#draw point on center of box (will be used in object tracking)
            cx = int((topLeft[0]+botRight[0])/2)
            cy = int((topLeft[1]+botRight[1])/2)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

			#Printing name on screen of aruco marker type and ID
            string = "{} ID: {}".format(aruco_type, ID)
            cv2.putText(img, string,(cx, cy+100), 
                        cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(ID))
    return [(cx, cy), topLeft, topRight, botRight, botLeft]
            

def main():
    cap = cv2.VideoCapture(0)
    while (1):
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [corners, ids, rejected] = detectAruco(img)
        [center, topLeft, topRight, botRight, botLeft] = displayAruco(img, corners, ids, rejected)
        cv2.imshow("image", img)
        if(cv2.waitKey(1)&0xFF==27):
            break
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()