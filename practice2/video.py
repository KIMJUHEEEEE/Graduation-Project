from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy import misc
import cv2
import numpy as np
import facenet
import detect_face
import os
import sys
import time
import pickle
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import dlib


modeldir = './model/20170511-185253.pb'
classifier_filename = './class/classifier.pkl'
npy='./npy'
pre_img="./pre_img"
COUNTER = 0
TOTAL = 0


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
	help="path to input video file")
ap.add_argument("-t", "--threshold", type = float, default=0.27,
	help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type = int, default=2,
	help="the number of consecutive frames the eye must be below the threshold")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Replace this with the path where you reside the file
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160
        
        HumanNames = os.listdir(pre_img)
        HumanNames.sort()
        listsize=len(HumanNames)
        ar = [0 for i in range(listsize)]
        print('Loading Modal')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)

        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_EXPOSURE, 1000)
        c = 0
        m=0
        p=0
        checkvideo=0
        print('Start Recognition')
        prevTime = 0
        cnt=0
        while cnt<60:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional
            img = cv2.flip(frame, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                k = 1
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                u1 = x + w
                u2 = y + h
                #cir = cv2.circle(img, (int(x + w / 2), int(y + h / 2)), 1, (0, 255, 255), 2)
                #print (x + w / 2, y + h / 2)
                #The below code is responsible for moving the mouse cursor in near real-time. Uncomment the below line to test it out.
                # m.moveTo(c1 * x + w / 2, c2 * y + h / 2, 0)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
                print(w*h)
                if w*h<7000 or w*h>15000:
                    checkvideo=1
            for (ex, ey, ew, eh) in eyes:
                k = 0
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            curTime = time.time()+1    # calc fps
            timeF = frame_interval
            if k == 1:
                print ("Blink")
                COUNTER+=1
            # The below snippet is responsible for mouse click events. At the moment this is would work but still needs enhancements
            # uncomment the below to enable mouse left click events.
            # m.click()
            if k == 0:
                print ("No Blink")
            if (c % timeF == 0):
                find_results = []

                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                frame = frame[:, :, 0:3]
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                print('Detected_FaceNum: %d' % nrof_faces)

                if nrof_faces > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]

                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((nrof_faces,4), dtype=np.int32)

                    for i in range(nrof_faces):
                        emb_array = np.zeros((1, embedding_size))

                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        # inner exception
                        if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                            print('Face is very close!')
                            continue

                        cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        #print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        # print("predictions")
                        #print(best_class_probabilities)
                        cnt+=1
                        ar[best_class_indices[0]]+=1
                        # print(best_class_probabilities)
                        if best_class_probabilities>0.53:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            #print('Result Indices: ', best_class_indices[0])
                            #print(HumanNames)
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    print(result_names)
                                    
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                            
                else:
                    print('Alignment Failure')
            
            cv2.imshow('Video', frame)
            if cnt==60:
                print(ar)
                print("COUNTER : ",COUNTER)
                for po in range(0,listsize):
                    if m<ar[po]:
                        m=ar[po]
                        p=po
                print('recognition : ',HumanNames[p])
                if COUNTER==0 or checkvideo==1:
                    print("NOT REAL")
                else:
                    print("REAL")
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        video_capture.release()
        cv2.destroyAllWindows()
