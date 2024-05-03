import cv2
import numpy as np

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

def detectAruco(img, aruco_type):
    arucoDict = cv2.aruco.Dictionary_get(aruco_lib[aruco_type])
    arucoParam = cv2.aruco.DetectorParameters_create()
    corners, ids, rej = cv2.aruco.detectMarkers(img, 
                                                  arucoDict, 
                                                  parameters=arucoParam)
    #print(corners)
    #print(ids)
    #print(rej)
    return [corners, ids, rej]

def displayAruco(img, corners, ids, rejected, aruco_type):
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
            cv2.putText(img, string,(cx, cy+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #print("[Inference] ArUco marker ID: {}".format(ID))
    return [(cx, cy), topLeft, topRight, botRight, botLeft]

def distanceNpose_estimation(img, cx, cy, markerSize, aruco_type, matrix_coefficients, distortion_coefficients):
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_type)
    parameters = cv2.aruco.DetectorParameters_create()
    distance = 0
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(img, cv2.aruco_dict,parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

    if corners:
        for i in range(0, len(ids)):
            rvec, tvec, points = cv2.aruco.estimatePoseSingleMarkers(corners[i], markerSize, matrix_coefficients,distortion_coefficients)
            cv2.aruco.drawAxis(img, matrix_coefficients, distortion_coefficients, rvec, tvec, 7)  

            distance = round(tvec[i][0][2], 2)
            #print("Distance: " + str(distance))

            String = "Distance: {}".format(distance)
            cv2.putText(img, String,(cx, cy-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return distance

def trackingAngle(center):
    center_angle = 90
    min_angle = 0
    max_angle = 180
    x = center[0]
    center_x = 640 / 2
    if x<(center_x+100) and x>(center_x-100):
        servo_angle = 90
    elif x < center_x:
        # Marker is on the left side of the screen
        servo_angle = (x / center_x) * (center_angle - min_angle) + min_angle
    else:
        # Marker is on the right side of the screen
        servo_angle = ((x - center_x) / (640 - center_x)) * (max_angle - center_angle) + center_angle
    if(servo_angle>=max_angle):
        servo_angle = max_angle
    if(servo_angle<=min_angle):
        servo_angle = min_angle
    return servo_angle