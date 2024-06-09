import numpy as np
from typing import Union
import os
import cv2
import glob
from scipy.optimize import fsolve


class Camera():
    """
    相机对象，记录相机的位置与朝向，是构造相机系统的基础。
    """
    def __init__(self, picture, data_path, chessboard_dir, location, eular_angle):
    eular_angle: np.ndarray = None
    camear_system: CameraSystem = None
    mesh = None
        """
        @brief 构造函数。
            1. 获取图片到自身的特征点（地面特征点）

        @param picture: 相机对应的图像。
        @param data_path: 相机的数据路径。
        @param chessboard_dir: 棋盘格图片的路径。
        @param location: 相机的空间位置（世界坐标）。
        @param eular_angle: 相机的欧拉角。
        """
        self.picture = picture
        self.picture.camera = self
        self.data_path = data_path
        self.chessboard_dir = chessboard_dir
        self.location = np.array(location)

        self.eular_angle = eular_angle
        self.axes = self.get_rot_matrix(eular_angle[0], eular_angle[1], eular_angle[2])

        self.DIM, self.K, self.D = self.get_K_and_D((4, 6), data_path + chessboard_dir)

        self.ground_feature_points = self.camera_to_world(picture.to_camera(picture.mark_board.reshape((-1,2)), 'L')).reshape((2,-1,3))
        self.feature_points = self.camera_to_world(picture.to_camera(picture.feature_point, 'L'))
        self.screen_feature_points = None

    def set_screen_frature_points(self, feature_point):
        """
        @brief 设置相机的屏幕特征点。
        @param args: 屏幕特征点。
        @return:
        """
        if isinstance(feature_point, list):
            self.feature_point.extend(feature_point)
        else:
            self.feature_point.append(feature_point)

    def get_rot_matrix(self, theta, gamma, beta) -> np.ndarray:
        """
        @brief 从欧拉角计算旋转矩阵。
        @param theta: 绕x轴的旋转角度。
        @param gamma: 绕y轴的旋转角度。
        @param beta: 绕z轴的旋转角度。
        @return: 旋转矩阵。
        """
        # 绕 x 轴的旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])

        # 绕 y 轴的旋转矩阵
        R_y = np.array([[np.cos(gamma), 0, np.sin(gamma)],
                        [0, 1, 0],
                        [-np.sin(gamma), 0, np.cos(gamma)]])

        # 绕 z 轴的旋转矩阵
        R_z = np.array([[np.cos(beta), -np.sin(beta), 0],
                        [np.sin(beta), np.cos(beta), 0],
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x
        return R

    def to_camera_system(self, *args):
        """
        @brief 调和映射，将相机上的点或网格映射到相机系统（视点）。
        @param args: 相机上的点或网格。
        @return:
        """
        assert self.camear_system is not None, "当前相机所属的相机系统未初始化。"
        if type(args[0]) in [list[np.ndarray], np.ndarray, list]:
            pass
        else:
            raise NotImplemented

    def to_picture(self, *args):
        """
        @brief 将相机上的点或网格映射到图像上。
        @param args: 相机上的点或网格
        @return:
        """
        if type(args[0]) in [list[np.ndarray], np.ndarray]:
            pass
        else:
            pass

    def projecte_to_self(self, point):
        """
        将点投影到相机球面上。
        @param points: 要投影的点。
        @return: 投影后的点。
        """
        v = points - self.location
        v = v/np.linalg.norm(v, axis=-1, keepdims=True)
        return v + self.location


    def to_screen(self, points):
        """
        将相机球面上的点投影到屏幕上。
        @param args: 相机球面上的点。
        @return:
        """
        screen = self.system.screen
        ret = screen.projecte_to_self(points, self.location, 1.0)
        return ret



    def camera_to_world(self, node):
        """
        @brief 把相机坐标系中的点转换到世界坐标系下
        """
        node = np.array(node)
        A = np.linalg.inv(self.axes.T)
        node = np.einsum('...j, jk->...k', node, A)
        node += self.location
        return node

    def get_K_and_D(self, checkerboard, imgsPath):
        CHECKERBOARD = checkerboard
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        _img_shape = None
        objpoints = []
        imgpoints = []
        images = glob.glob(imgsPath + '/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            #ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        DIM = _img_shape[::-1]
        return DIM, K, D
