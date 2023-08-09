import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("src/images/person4_1.png")
img2 = cv2.imread("src/images/person4_1.png")

orb = cv2.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)
# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
 table_number = 6, # 12
 key_size = 12, # 20
 multi_probe_level = 1) #2
search_params = dict(checks=50) # or pass empty dictionary
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
        
    print(b)
    print(len(des2))
    print(b/len(des2))
    draw_params = dict(matchColor = (0,255,0),
    singlePointColor = (255,0,0),
    matchesMask = matchesMask,
    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()
else:
    print("No matches")

