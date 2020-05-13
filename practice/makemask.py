import os
import cv2


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error : Creating directory.'+directory)


cnt=0
for fol in os.walk('/home/juhee/Graduation-Project/practice/pre_img'):
    cnt+=1
    if cnt==1: continue
    l=fol[0]
    l=l+'_mask'
    createFolder(l)
    for pi in os.walk(fol[0]):
        b=len(pi[2])
        for ch in range(0,b):
            s=pi[2][ch]
            a='mask_'+s
            Img = cv2.imread(fol[0]+'/'+s)
            for x in range(95,182):
                for y in range(0,182):
                    Img[x,y]=255
            cv2.imwrite(os.path.join(l,a),Img)
