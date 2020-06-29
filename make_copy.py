import os
import cv2

cnt=0
for fol in os.walk('/home/juhee/Graduation-Project/practice2/pre_img'):
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
            l=a+"/"
            for cnt in range(1,501):
                sa=str(cnt)
                cv2.imwrite(l+sa+s,Img)
 
            



