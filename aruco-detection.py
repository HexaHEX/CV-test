from __future__ import division
import math

import cv2.aruco
import numpy as np
from itertools import zip_longest

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
board = cv2.aruco.CharucoBoard_create(5, 5, 0.032, 0.024, dictionary)
img = board.draw((1000, 1000))
cv2.imwrite('charuco_5x5-1000_32x28_0.032-0.024.png', img)
img = board.draw((1000, 1000))
cv2.imwrite('charuco_5x5-1000_32x28_0.032-0.024_large.png', img)



class CameraModel(object):
    def __init__(self, frame_width, frame_height, field_of_view_x, camera_matrix, dist_coeffs, new_camera_matrix):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.field_of_view_x = field_of_view_x
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.new_camera_matrix = new_camera_matrix
        self.principal_point_x = camera_matrix[0][2]
        self.principal_point_y = camera_matrix[1][2]
        
    def _calc_angle(self, pixels_from_center):
        return math.atan(pixels_from_center * math.tan(self.field_of_view_x / 2) / (self.frame_width / 2))

    def px2meters(self, x, y, altitude, pitch=0, roll=0):
        x_from_principal_point = x - self.principal_point_x
        y_from_principal_point = y - self.principal_point_y
        
        x_angle = -self._calc_angle(x_from_principal_point) + pitch
        y_angle =  self._calc_angle(y_from_principal_point) + roll
        
        x_meters_from_center = -altitude * math.tan(x_angle)
        y_meters_from_center =  altitude * math.tan(y_angle)
        
        return x_meters_from_center, y_meters_from_center

FRAME_WIDTH = 1000
FRAME_HEIGHT = 800
FOV = math.radians(140)

ALTITUDE = 0.5

# Coeffs for RPi Camera (G)
ORIG_CAMERA_MATRIX = np.array([[ 851.97704662,    0.        ,  831.2243264 ],
                               [   0.        ,  855.15220764,  563.7543571 ],
                               [   0.        ,    0.        ,    1.        ]])
ORIG_CAMERA_WIDTH = 1640
ORIG_CAMERA_HEIGHT = 1232
DIST_COEFFS = np.array([[  2.15356885e-01,  -1.17472846e-01,  -3.06197672e-04,
                          -1.09444025e-04,  -4.53657258e-03,   5.73090623e-01,
                          -1.27574577e-01,  -2.86125589e-02,   0.00000000e+00,
                           0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                           0.00000000e+00,   0.00000000e+00]])

camera_matrix = ORIG_CAMERA_MATRIX * FRAME_WIDTH / ORIG_CAMERA_WIDTH
camera_matrix[2, 2] = 1.0

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                       DIST_COEFFS,
                                                       (FRAME_WIDTH, FRAME_HEIGHT),
                                                       1,
                                                       (FRAME_WIDTH, FRAME_HEIGHT))

camera_model = CameraModel(FRAME_WIDTH, FRAME_HEIGHT, FOV, camera_matrix, DIST_COEFFS, new_camera_matrix)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
board = cv2.aruco.CharucoBoard_create(5, 5, 0.032, 0.024, dictionary)

charuco_board_d = {i: corner for i, corner in enumerate(board.chessboardCorners)}

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)



while(True):
    _, frame = capture.read()
    gray_frame = frame
    undist_frame = cv2.undistort(gray_frame, camera_matrix, DIST_COEFFS, newCameraMatrix=new_camera_matrix)
    aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(gray_frame, dictionary)
    
    if len(aruco_corners) > 0:
        cv2.aruco.drawDetectedMarkers(gray_frame, aruco_corners, aruco_ids)
        charuco_detected, charuco_corners, charuco_ids_in_numpy = cv2.aruco.interpolateCornersCharuco(aruco_corners,
                                                                                                      aruco_ids,
                                                                                                      gray_frame,
                                                                                                      board,
                                                                                                      None,
                                                                                                      None,
                                                                                                      camera_matrix,
                                                                                                      DIST_COEFFS)
        
        if charuco_detected:
            charuco_ids = [id_in_numpy[0] for id_in_numpy in charuco_ids_in_numpy]
            
            undist_charuco_corners = cv2.undistortPoints(charuco_corners, camera_matrix, DIST_COEFFS, P=new_camera_matrix)
            
            assert len(charuco_ids) == len(undist_charuco_corners)
            charuco_corners_d = {id_: coordinates for id_, coordinates in zip(charuco_ids, undist_charuco_corners)}
            
            for id_, coordinates in charuco_corners_d.items():
                coordinates_meters = camera_model.px2meters(coordinates[0][0], coordinates[0][1], ALTITUDE)
                print ("%s\t%s\t%s" % (str(id_), str(charuco_board_d[id_]), str(coordinates_meters)) )
            print ("--------------------------")
            
            cv2.aruco.drawDetectedCornersCharuco(undist_frame, undist_charuco_corners, charuco_ids_in_numpy)
            pose_detected, pose_rvec, pose_tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners,
                                                                                     charuco_ids_in_numpy,
                                                                                     board,
                                                                                     camera_matrix,
                                                                                     DIST_COEFFS)
           
            if pose_detected:
                print (pose_rvec)
                print (pose_tvec)
                print ("===========================")
                cv2.aruco.drawAxis(undist_frame, camera_matrix, DIST_COEFFS, pose_rvec, pose_tvec, 0.1)




    u0 = int(undist_frame.shape[0]/4)
    u1 = int(undist_frame.shape[1]/4)
    small_frame = cv2.resize(undist_frame, (u1, u0))
    # crop = gray[(gray_frame.shape[0]/2 - 150):(gray_frame.shape[0]/2 + 150),
    #             (gray_frame.shape[1]/2 - 150):(gray_frame.shape[1]/2 + 150)]
    cv2.imshow('frame', undist_frame)
    #print(ALTITUDE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
