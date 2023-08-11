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
from sensor_msgs.msg import PointCloud2,Image,CameraInfo

zed=None
init_params=None
runtime_parameters=None
fx = None
fy = None
cx = None
cy = None

def get_depth(data):
    global final,fx,fy,cx,cy
    img = CvBridge().imgmsg_to_cv2(data, desired_encoding='passthrough')

    rgba  = struct.unpack('I', struct.pack('BBBB', 255, 255, 255, 255))[0]
    points=[]

    print(final.shape)
    for row in range(final.shape[0]):
        for column in range(final.shape[1]):
            if sobelxy[row][column] == 0:
                depth=img[row,column]
                x=(column-cx)*depth/fx
                y=(row-cy)*depth/fy
                points.append([x,depth,10,rgba]) 

    publish_point_cloud(points)  
    

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

final=None
def callback(data):
    global sobelxy
    img = CvBridge().imgmsg_to_cv2(data, desired_encoding='passthrough')
    img_blur = cv2.GaussianBlur(img, (3,3), 0) 

    # sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    final = img_blur

def get_cam_matrix(data):
    mat = np.array(data.K)
    global fx,fy,cx,cy
    fx = mat[0]
    fy = mat[4]
    cx = mat[2]
    cy = mat[5]

    
def main():
    
    rospy.init_node('costmap_publisher', anonymous=True)
    rospy.Subscriber("/zed2i/drivable_region",Image,callback)
    rospy.Subscriber("zed2i/zed_node/depth/depth_registered",Image,get_depth)
    rospy.Subscriber("/zed2i/zed_node/depth/camera_info",CameraInfo,get_cam_matrix)
    rospy.spin()
        
if __name__ == "_main_":
    main()