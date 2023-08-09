#!/usr/bin/env python3
import DynamicObstacles
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
import pandas as pd
from cv_bridge import CvBridge


prev_image = None
# centre : centre of the image (tuple of the form(x,y))
# image : the detected image
    
def callback(data):
    global tracked
    global do_obj
    
    len_tracked = len(tracked.index)
    img = CvBridge().imgmsg_to_cv2(data, desired_encoding='passthrough')
    
    cv2.imshow("haha",img)
    cv2.waitKey(0)
    
    detections = do_obj.get_detections(img,0.3)
    len_detections = len(detections.index)
    
    rospy.loginfo(detections)
    
    # If no tracked objects are yet there, then add the newly detected objects
    if len_tracked==0 and len_detections!=0:
        for index,detection in detections.iterrows():
            imgcrop = img[int(np.round(detection["ymin"])):int(np.round(detection["ymax"])),int(np.round(detection["xmin"])):int(np.round(detection["xmax"]))]
            centre = ((detection["xmin"]+detection["xmax"])/2,(detection["ymin"]+detection["ymax"])/2)
            tracked = pd.concat([tracked,pd.DataFrame({"image":[imgcrop], "centre":[centre]})])
    
     
    # If tracked objects are already there, then feature matching is performed and 
    # new objects are added, existing objects are updated
    elif len_tracked!=0 and len_detections!=0:
        for index,detection in detections.iterrows():
            imgcrop = img[int(np.round(detection["ymin"])):int(np.round(detection["ymax"])),int(np.round(detection["xmin"])):int(np.round(detection["xmax"]))]
            centre = ((detection["xmin"]+detection["xmax"])/2,(detection["ymin"]+detection["ymax"])/2)
            for index,tracked_obj in tracked.iterrows():
                if do_obj.feature_match_found(tracked_obj["past"],imgcrop):
                    
                    # TODO: 
                    #   Calculate direction and velocity of motion
                    tracked.loc[index,["image","centre"]] = [imgcrop,centre]
           
    
    # If no object is detected, then no objects are tracked
    else:
        tracked = pd.DataFrame({"object":[],"past":[],"present":[],"vel":[]})
    
    
    for index,detection in detections.iterrows():
        imgcrop = img[int(np.round(detection["ymin"])):int(np.round(detection["ymax"])),int(np.round(detection["xmin"])):int(np.round(detection["xmax"]))]
        do_obj.feature_match_found(imgcrop,img)
    
      
if __name__ == '__main__':
    
    tracked = pd.DataFrame({"image":[],"centre":[]})
    do_obj = DynamicObstacles.DynamicObstacles("/home/kallrax/abhiyaan_ws/misc_ws/src/obstacle_detection/src/dynamic_obstacles/yolov5","/home/kallrax/abhiyaan_ws/misc_ws/src/obstacle_detection/src/dynamic_obstacles/models/best.pt")

    
    #Initialising node
    rospy.init_node('trajectory_prediction_node', anonymous=True)
    rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color",Image, callback) 

    rospy.spin()