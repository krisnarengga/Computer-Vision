import cv2
import numpy as np

MIN_MATCH_COUNT=20

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

obj_name = ['chili','milk','coca cola','car','motorcycle','bus']
train_Img = []
train_KP = []
train_Desc = []
for i in range(0,6):
    obj_str = obj_name[i]+'.jpg'
    img = cv2.imread(obj_str,0)
    img_resize = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    train_Img.append(img_resize)    
    trainKP,trainDesc=detector.detectAndCompute(train_Img[i],None)
    train_KP.append(trainKP)
    train_Desc.append(trainDesc)
    

cam=cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)

    idx = -1
    for i in range(0,6):
        matches=flann.knnMatch(queryDesc,train_Desc[i],k=2)
        goodMatch=[]
        for m,n in matches:
            if(m.distance<0.8*n.distance):
                goodMatch.append(m)

        if(len(goodMatch)>=MIN_MATCH_COUNT):
            idx = i
            break
        else:
            idx = -1
    			
    if(idx!=-1 and len(goodMatch)>=MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(train_KP[idx][m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,5.0)
        h,w=train_Img[idx].shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),2)
	cv2.putText(QueryImgBGR, 'This is '+obj_name[idx], (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
        
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()
