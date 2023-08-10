#!/usr/bin/env python3

import rospy
import pyzed.sl as sl
import math
import numpy as np
import sys
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import PointCloud2,Image

zed=None
init_params=None
runtime_parameters=None

def init_zed():
    # TODO:
    #   HOW TO USE ZED SDK ALONG WITH ZED ROS WRAPPER
    
    global zed,init_params,runtime_parameters
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("HAHA")
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.texture_confidence_threshold = 100
    

def get_depth(x,y):
    global zed,init_params,runtime_parameters
    
    depth = sl.Mat()
    point_cloud = sl.Mat()

    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m
    
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve depth map. Depth is aligned on the left image
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # Get and print distance value in mm at the center of the image
        # We measure the distance camera - object using Euclidean distance
        err, point_cloud_value = point_cloud.get_value(x, y)



        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                             point_cloud_value[1] * point_cloud_value[1] +
                             point_cloud_value[2] * point_cloud_value[2])

        point_cloud_np = point_cloud.get_data()
        point_cloud_np.dot(tr_np)

        if not np.isnan(distance) and not np.isinf(distance):
            return distance
        else:
            # If the point is too close to the camera
            return 0
        sys.stdout.flush()
def publish_point_cloud(points):
    point_cloud_pub = rospy.Publisher("segmentation/pointcloud", PointCloud2, queue_size=10)
    
    header = Header()
    header.frame_id = "base_link"
    pc2 = point_cloud2.create_cloud(header, [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgba', 12, PointField.UINT32, 1),
            ], points)
    pc2.header.stamp = rospy.Time.now()
    point_cloud_pub.publish(pc2)

def callback(data):
    img = CvBridge().imgmsg_to_cv2(data, desired_encoding='passthrough')
    img_blur = cv2.GaussianBlur(img, (3,3), 0) 

    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    rgba  = struct.unpack('I', struct.pack('BBBB', 255, 255, 255, 255))[0]
    points=[]
    for row in range(sobelxy.shape[0]):
        for column in range(sobelxy.shape[1]):
            if sobelxy[row][column] == 255:
                depth=get_depth(column,row)
                [points.append([column,depth,i,rgba]) for i in range(50)]

    publish_point_cloud(points)
    

    
def main():
    
    rospy.init_node('costmap_publisher', anonymous=True)
    init_zed()
    rospy.Subscriber = rospy.Subscriber("/zed2i/drivable_region",Image,callback)
    rospy.spin()
        
if __name__ == "__main__":
    main()