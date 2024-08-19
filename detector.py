import os
import cv2 
import numpy as np  

def empty(a):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",216,255,empty)
cv2.createTrackbar("Threshold2","Parameters",222,255,empty)



def color_segmentation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for red and yellow
    red_lower = np.array([0, 40, 20])
    red_upper = np.array([20, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower, red_upper)


    red_lower = np.array([160, 40, 20])
    red_upper = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, red_lower, red_upper)
    
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    blue_lower = np.array([90, 20, 100])
    blue_upper = np.array([143, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    white_lower = np.array([0,0,200])
    white_upper = np.array([180,255,255])
    white_mask = cv2.inRange(hsv,white_lower,white_upper)

    # Combine masks
    mask = cv2.bitwise_or(red_mask1, red_mask2)
    #mask = cv2.bitwise_or(mask, yellow_mask)
    mask = cv2.bitwise_or(mask, blue_mask)
    mask = cv2.bitwise_or(mask,white_mask)
    color_result = cv2.bitwise_and(image, image, mask=mask)

    return color_result


def drawSign(imgCnt,cnt,approx,area,x,y,w,h,name):
    cv2.drawContours(imgCnt,cnt,-1,(255,255,0),3)
    cv2.rectangle(imgCnt,(x,y),(x+w,y+h),(0,255,0),3)

    pointsText = "Points:" + str(len(approx))
    areaText = "Area:" + str(int(area))
    signText = name
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1

    text_size,_ = cv2.getTextSize(pointsText,font,fontScale,fontThickness)
    tw,th=text_size
    cv2.rectangle(imgCnt,(x+w+19,y+21),(x+w+21+tw,y+19-th),(0,0,0),-1)
    cv2.putText(imgCnt,pointsText ,(x+w+20,y+20),font,fontScale,(0,255,0),fontThickness)

    text_size,_ = cv2.getTextSize(areaText,font,fontScale,fontThickness)
    tw,th=text_size
    cv2.rectangle(imgCnt,(x+w+19,y+46),(x+w+21+tw,y+44-th),(0,0,0),-1)
    cv2.putText(imgCnt,areaText ,(x+w+20,y+45),font,fontScale,(0,255,0),fontThickness)

    text_size,_ = cv2.getTextSize(signText,font,fontScale,fontThickness)
    tw,th=text_size
    cv2.rectangle(imgCnt,(x+9,y+16),(x+11+tw,y+14-th),(0,0,0),-1)
    cv2.putText(imgCnt,signText ,(x+10,y+15),font,fontScale+0.1,(0,255,0),fontThickness)
    return imgCnt


def getCountour(raw_img,img,pastSignLocs):

    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    imgContour = raw_img.copy()

    for p in pastSignLocs:
        p[-1] = False

    for it,cnt in enumerate(contours):

        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.02 * peri, True)
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(approx)
        ratio = 0
        if w > h:
            ratio = w/h
        else:
            ratio = h/w

        
        signMoveThr = 10
        signAreaThr = 30
        oldSign = False
        for p in pastSignLocs:
            ps = p[0]
            px = p[1]
            py = p[2]
            pa = p[3]

            if abs(x-px) < signMoveThr and abs(y-py) < signMoveThr and abs(pa-area)<signAreaThr and p[-1] == False:
                # old sign 
                p[-1] = True  
                oldSign=True
                drawSign(imgContour,cnt,approx,area,px,py,w,h,ps)
                break

        if oldSign == False and ratio < 1.5 and area > 1800 and hierarchy[0,it,3] == -1:

            # make a template and then compare
            bestScore = 0
            bestScoreSign = ""
            match_threshold = 0.1

            for templateTuple in signTemplateImages:
                template = cv2.cvtColor(cv2.GaussianBlur(cv2.resize(templateTuple[0],(w,h)),(7,7),1),cv2.COLOR_BGR2GRAY)
                template = cv2.Canny(template,216,222)

                cropped = img[y:y+h,x:x+w]
                
                res = cv2.matchTemplate(cropped, template, cv2.TM_CCORR_NORMED)
                _,max,_,_ = cv2.minMaxLoc(res)
                
                #print(filename," ==> ",max)
                if max > bestScore :
                    
                    bestScore = max
                    bestScoreSign = templateTuple[1]
                    
            if bestScore > match_threshold:
                pastSignLocs.append([bestScoreSign,x,y,area,True])
                drawSign(imgContour,cnt,approx,area,x,y,w,h,bestScoreSign)

    for p in pastSignLocs:
        tmp = []
        if p[-1] == True:
            tmp.append(p)
        del pastSignLocs
        pastSignLocs = tmp
    print(len(pastSignLocs))
    return imgContour


frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)

pastSigns = []

signTemplateImages = []
for filename in os.listdir("signs"):
    t = cv2.imread(os.path.join("signs",filename))
    if t is not None:
        signTemplateImages.append((t,filename))


while(True):

    success, img = cap.read()
    #img = cv2.imread("left_right.png")

    img = color_segmentation(img)
    imgBlur = cv2.GaussianBlur(img,(7,7),1)
    imgCvt = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)

    thr1= cv2.getTrackbarPos("Threshold1","Parameters")
    thr2= cv2.getTrackbarPos("Threshold2","Parameters")
    
    imgCanny = cv2.Canny(imgCvt,thr1,thr2)
    imgResultContour = getCountour(img,imgCanny,pastSigns)

    cv2.imshow("img canny",imgCanny)
    cv2.imshow("img contour",imgResultContour)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows() 