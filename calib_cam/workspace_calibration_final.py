import cv2
#assert cv2.__version__[0] == '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

import matplotlib.pyplot as plt


if __name__ == "__main__":

    omnidir=True
    rectilinear=False

    images = glob.glob('calib_new_photos/*.jpg')
    
    CHECKERBOARD = (10,15)#(6,9)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT, 30000, 0.00001)  #30, 0.1)
    calibration_flags = 0 #cv2.omnidir.CALIB_USE_GUESS + cv2.omnidir.CALIB_FIX_SKEW + cv2.omnidir.CALIB_FIX_CENTER


    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    files_usable = []
    files_unusable = []
    i=0
    for fname in images:
        print(i,fname)
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        # print(ret)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria) #this function mutates corners
            imgpoints.append(corners)
            files_usable.append(fname)

            # cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
            # cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
            # cv2.imshow('finalImg', img)
            # cv2.waitKey(500)
        else:
            files_unusable.append(fname)
        i+=1
        # print('---')

    N_OK = len(objpoints)


    if omnidir:
        rms, k, xi, d, rvecs, tvecs, idx  =  cv2.omnidir.calibrate(
                    objectPoints=objpoints, 
                    imagePoints=imgpoints, 
                    size=gray.shape[::-1], 
                    K=None, xi=None, D=None,
                    flags=calibration_flags,
                    criteria=subpix_criteria)

        idx_list=list(idx.squeeze())

        # new_width, new_height = 1000,500
        # Knew = np.array([[new_width/3.1415, 0, 0],
        #                [0, new_height/3.1415, 0],
        #                [0, 0, 1]])
        # #if rectify perspective
        # # Knew = np.array([[new_width / 4, 0, new_width / 2],
        # #                  [0, new_height / 4, new_height / 2],
        # #                  [0, 0, 1]], dtype=np.float32)
        # for i, fname in enumerate(images):
        #     print(i,fname)
        #     img = cv2.imread(fname)
        #     #RECTIFY_CYLINDRICAL
        #     #RECTIFY_STEREOGRAPHIC
        #     undistorted = cv2.omnidir.undistortImage(img,K,D,XI,cv2.omnidir.RECTIFY_CYLINDRICAL,Knew = Knew, new_size = [new_width,new_height])
        #     cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
        #     cv2.imshow('finalImg', undistorted)
        #     cv2.waitKey()
        #     break
        ## https://docs.opencv.org/4.x/dd/d12/tutorial_omnidir_calib_main.html


        # print('--',type(xi),len(objpoints),len(rvecs),idx, idx.shape,imgpoints[0].shape)
        print('=='*30)
        mean_error = 0
        for i in range(len(files_unusable)):
            print('Failed to detect   ',files_unusable[i])
        for i in range(len(objpoints)):
            if i not in idx:
                print('Failed to calibrate',files_usable[i])

        for i in range(len(idx_list)):
            imgpoints2, _ = cv2.omnidir.projectPoints(objpoints[int(idx_list[i])],rvecs[i],tvecs[i],k,float(xi),d)     #objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            # error = cv2.norm(imgpoints[int(idx_list[i])], imgpoints2.transpose([1,0,2]), cv2.NORM_L2)/len(imgpoints2)
            error = np.linalg.norm(imgpoints[int(idx_list[i])] - imgpoints2.transpose([1,0,2]), axis=2)
            error = np.sqrt(np.mean(error*error))
            print(files_usable[int(idx_list[i])],'\t', error)
            mean_error += error
        
        print( "total MRMS error: {}".format(mean_error/len(objpoints)) )

        print("Found " + str(N_OK) +"/"+ str(len(images)) + " chessboard corners")
        print("Used "+str(len(idx_list))+" images for OMNIDIR calibration")
        print("RMS value: " + str(rms))
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(k.tolist()) + ")")
        print("XI=np.array(" + str(xi.tolist()) + ")")
        print("D=np.array(" + str(d.tolist()) + ")")


        img = cv2.imread(files_usable[int(idx_list[0])])
        cv2.namedWindow('PointsImg', cv2.WINDOW_NORMAL)
        
        for i in range(len(idx_list)):
            points = imgpoints[int(idx_list[i])].squeeze() #(150,1,2) -> (150,2)
            for p in points:
                cv2.circle(img,center=(int(p[0]),int(p[1])),radius=2,color=(0,0,255),thickness=-1)
        cv2.imshow('PointsImg', img)    
        cv2.waitKey()





    elif rectilinear:
        # flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL

        # flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_TILTED_MODEL

        flags=None

        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints=objpoints, 
                    imagePoints=imgpoints, 
                    imageSize=gray.shape[::-1], cameraMatrix=None, distCoeffs= None, flags=flags)

        print("Found " + str(N_OK) + " valid images for OMNIDIR calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(cameraMatrix.tolist()) + ")")
        print("D=np.array(" + str(distCoeffs.tolist()) + ")")
        print('retval: ',retval)
