import os
import cv2

cnt=0
for fol in os.walk('/home/juhee/Facenet/pre_img'):
    cnt+=1
    if cnt==1: continue
    a=fol[0]
    for pi in os.walk(a):
        b=len(pi[2])
        for ch in range(0,b):
            l=""
            s=pi[2][ch]
            l=a+"/"
            l=l+s
            Img = cv2.imread(l)
            FImg = cv2.flip(Img,1)
 #           grayImg = cv2.imread(l, cv2.IMREAD_GRAYSCALE) 
 #           brightImg = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
 #           darkImg = cv2.imread(l,cv2.IMREAD_GRAYSCALE)
 #           for x in range(0,182):
 #               for y in range(0,182):
 #                   if(brightImg[x,y]>240):
 #                       brightImg[x,y]=255
 #                      continue
 #                   brightImg[x,y]+=10
 #            
 #                   
 #           for x in range(0,182):
 #               for y in range(0,182):
 #                   if(brightImg[x,y]<100):
 #                       brightImg[x,y]=0
 #                       continue
 #                   brightImg[x,y]-=10        
            l=a+"/"
 #           cv2.imwrite(l+"copy"+s, grayImg) 
            cv2.imwrite(l+"flip"+s,FImg)
 #           cv2.imwrite(l+"bright"+s,brightImg)
            



