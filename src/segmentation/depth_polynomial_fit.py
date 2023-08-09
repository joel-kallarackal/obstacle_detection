import cv2
import numpy as np
import matplotlib.pyplot as plt

img_ground = cv2.imread("images/depth_ground1.jpeg")
img_ground_gray = cv2.cvtColor(img_ground,cv2.COLOR_BGR2GRAY)

def get_ground_curve_from_depth_image(depth_img_ground):
    '''
        Fits a polynomial to the depth image of the ground
    '''
    m = depth_img_ground.shape[0] # m is the number of rows in the image
    n = depth_img_ground.shape[1] # n is the number of columns in the image
    
    X = np.array([x for x in range(m)])
    Y = depth_img_ground.T
    a=2
    
    coefficients = np.polyfit(X, Y[a], 4)
    poly = np.poly1d(coefficients)
    print(poly(X))
    
    '''
        TODO:
        Run RANSAC and find out the best fit curve
    '''
    
    
    
    x_curve = np.array([x for x in range(m)])
    y_curve = poly(x_curve)
    
    # Visualization
    plt.scatter(X, Y[a], label='Data Points')
    plt.plot(x_curve, y_curve, 'r', label='Fitted Curve')
    plt.legend()
    plt.show()
    
    return poly
   
def segment_ground_plane(depth_img,ground_polynomial):
    '''
        Ground polynomial is the polynomial fitted to the depth image of the ground
        This function applies the ground poynomial to the depth image and classfies each pixel
        as ground or not ground
    '''
    
    m = depth_img.shape[0] # m is the number of rows in the image
    n = depth_img.shape[1] # n is the number of columns in the image
    Y_ground = ground_polynomial(np.array([x for x in range(m)])) 
    depth_image_transpose = depth_img.T.copy()
    for i in range(n):
        
        Y_img = ground_polynomial(depth_img.T[i])
        err = np.abs(Y_img-Y_ground)/255
        depth_image_transpose[i][err<=0.1] = 255
        depth_image_transpose[i][err>0.1] = 0
    
    cv2.imshow("segmented image",depth_image_transpose.T)
    cv2.waitKey(0)
   
img_depth = cv2.GaussianBlur(cv2.cvtColor(cv2.imread("images/depth2.jpeg"),cv2.COLOR_RGB2GRAY),(7,7),0)
segment_ground_plane(img_depth,get_ground_curve_from_depth_image(img_ground_gray))
# get_ground_curve_from_depth_image(img_ground_gray)  

