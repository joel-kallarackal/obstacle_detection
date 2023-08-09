import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

class DynamicObstacles:
    def __init__(self,yolov5_path : str,model_path : str):
        """
        Args:
            yolov5_path (string): path to yolov5 repository cloned from https://github.com/ultralytics/yolov5/
            model_path (string): path to .pt file
        """
        self.model = torch.hub.load(yolov5_path, 'custom', path=model_path, source='local') 
    
    def get_midpoints_of_detections(self,img,confidence):
        results = self.model(img)
        filtered_results = results.pandas().xyxy[0][results.pandas().xyxy[0]['confidence']>confidence]
        
        for i in range(len(filtered_results)):
            x_mid = round((filtered_results.iloc[i]['xmin']+filtered_results.iloc[i]['xmax'])/2)
            y_mid = round((filtered_results.iloc[i]['ymin']+filtered_results.iloc[i]['ymax'])/2)
        
        return (x_mid,y_mid)
    
    def get_detections(self,img,confidence):
        """
        Returns a pandas.DataFrame which contains all the detctions
        """
        results = self.model(img)
        return results.pandas().xyxy[0][results.pandas().xyxy[0]['confidence']>confidence]
    
    
    def feature_match_found(self,past_image,present_image,thresh=0.1):
        """
        Returns true if the two images represent the same object in two different frames
        """
        
        orb = cv2.ORB_create()
        
        # find the keypoints with ORB
        kp1 = orb.detect(past_image,None)
        kp2 = orb.detect(present_image,None)
        
        # compute the descriptors with ORB
        kp1, des1 = orb.compute(past_image, kp1)
        kp2, des2 = orb.compute(present_image, kp2)
        
        
        
        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
        table_number = 6, 
        key_size = 12, 
        multi_probe_level = 1) 
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches=None
        if(des1 is not None and len(des1)>2 and des2 is not None and len(des2)>2):
            matches = flann.knnMatch(des1,des2,k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            
            # ratio test as per Lowe's paper
            a=0
            b=0
            for match in matches:
                try:
                    if match[0].distance < 0.7*match[1].distance:
                        b+=1
                        matchesMask[a]=[1,0]
                except:
                    continue
            
            #Check if valid match is found
            if b/len(matches)>thresh:
                return True
            else:
                return False
        else:
            return False
        
    
        