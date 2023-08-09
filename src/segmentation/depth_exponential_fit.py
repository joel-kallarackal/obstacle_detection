import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

img_ground = cv2.imread("images/depth_ground3.jpeg")
img_ground_gray = cv2.cvtColor(img_ground,cv2.COLOR_BGR2GRAY)

def exponential_func(x, a, b, c, d, e):
    return a*np.exp(-b*x)+c*np.exp(-d*x)

def get_ground_curve_from_depth_image(depth_img_ground):
    '''
        Fits a polynomial to the depth image of the ground
    '''
    m = depth_img_ground.shape[0] # m is the number of rows in the image
    n = depth_img_ground.shape[1] # n is the number of columns in the image
    a=5
    X = np.array([x for x in range(m)])
    Y = depth_img_ground.T
    X1 = np.array([x for x in range(500)])
    popt, pcov = curve_fit(exponential_func, X, Y[a]) 
    # popt contain the optimized parameters of the exponential function
    # pcov will contain the estimated covariance of popt
    
    plt.scatter(X, Y[a], label='Original data')
    plt.plot(X1, exponential_func(X1, *popt), 'r-', label='Fitted curve')
    plt.legend()
    plt.show()
    
    '''
        TODO:
        Run RANSAC and find out the best fit curve
    '''
    
    return popt
   
def segment_ground_plane(depth_img,popt):
    '''
        Ground polynomial is the polynomial fitted to the depth image of the ground
        This function applies the ground poynomial to the depth image and classfies each pixel
        as ground or not ground
    '''
    
    m = depth_img.shape[0] # m is the number of rows in the image
    n = depth_img.shape[1] # n is the number of columns in the image
    Y_ground = exponential_func(np.array([x for x in range(m)]), *popt)
    depth_image_transpose = depth_img.T.copy()
    for i in range(n):
        
        Y_img = exponential_func(depth_img.T[i], *popt)
        err = np.abs(Y_img-Y_ground)/255
        depth_image_transpose[i][err<=0.5] = 255
        depth_image_transpose[i][err>0.5] = 0
    
    cv2.imshow("segmented image",depth_image_transpose.T)
    cv2.waitKey(0)
   
img_depth = cv2.GaussianBlur(cv2.cvtColor(cv2.imread("images/depth3.jpeg"),cv2.COLOR_RGB2GRAY),(1,1),0)
segment_ground_plane(img_depth,get_ground_curve_from_depth_image(img_ground_gray))
# get_ground_curve_from_depth_image(img_ground_gray)  

