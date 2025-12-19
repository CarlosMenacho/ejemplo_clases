import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

import cv2
import numpy as np


class ArucoDetector(Node):

    def __init__(self):
        super().__init__("cv_aruco")

        self.bridge = CvBridge()

        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        self.sub_img = self.create_subscription(Image, "/camera/image_raw",
                                                self.img_callback, 10)

        self.center_img = None

    def img_callback(self, msg):

        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self.center_img is None:
            height, width = cv_img.shape[:2]
            self.center_img = np.array([width / 2.0, height / 2.0])
            print(f"centro {self.center_img}")

        corners, ids, rej = cv2.aruco.detectMarkers(
            cv_img, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(cv_img, corners, ids)
            markers_corners = corners[0][0]
            marker_center = np.mean(markers_corners, axis=0)

            error = marker_center = self.center_img

            cv2.circle(cv_img, tuple(marker_center.astype(int)), 5,
                       (0, 255, 0), -1)

            center_msg = Point()
            center_msg.x = float(marker_center[0])
            center_msg.y = float(marker_center[1])
            center_msg.z = 0.0

        cv2.imshow("robot_vis", cv_img)
        cv2.waitKey(1)
        # print(cv_img.shape)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node=node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
