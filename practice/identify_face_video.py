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



def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 

def count() :
    args = vars(ap.parse_args())
    EYE_AR_THRESH = args['threshold']
    EYE_AR_CONSEC_FRAMES = args['frames']
    global COUNTER
    global TOTAL
    # initialize the frame counters and the total number of blinks
#    COUNTER = 0
 #   TOTAL = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
 #   print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
 
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # start the video stream thread
#    print("[INFO] starting video stream thread...")
#    print("[INFO] print q to quit...")
#    time.sleep(1.0)
    
    # loop over frames from the video stream
#    while True:
    	# if this is a file video stream, then we need to check if
    	# there any more frames left in the buffer to proces    
    	# grab the frame from the threaded video file stream, resize
    	# it, and convert it to grayscale
    	# channels)
    #frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
    	# determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy
    	# array
    	shape = predictor(gray, rect)
    	shape = face_utils.shape_to_np(shape)

    	# extract the left and right eye coordinates, then use the
    	# coordinates to compute the eye aspect ratio for both eyes
    	leftEye = shape[lStart:lEnd]
    	rightEye = shape[rStart:rEnd]
    	leftEAR = eye_aspect_ratio(leftEye)
    	rightEAR = eye_aspect_ratio(rightEye)

    	# average the eye aspect ratio together for both eyes
    	ear = (leftEAR + rightEAR) / 2.0

    	# compute the convex hull for the left and right eye, then
    	# visualize each of the eyes
    	leftEyeHull = cv2.convexHull(leftEye)
    	rightEyeHull = cv2.convexHull(rightEye)
    	cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    	cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    	# check to see if the eye aspect ratio is below the blink
    	# threshold, and if so, increment the blink frame counter
    	if ear < EYE_AR_THRESH:
    		COUNTER += 1

    	# otherwise, the eye aspect ratio is not below the blink
    	# threshold
    	else:
    		# if the eyes were closed for a sufficient number of
    		# then increment the total number of blinks
    		if COUNTER >= EYE_AR_CONSEC_FRAMES:
    			TOTAL += 1

    		# reset the eye frame counter
    		#COUNTER = 0

    	# draw the total number of blinks on the frame along with
    	# the computed eye aspect ratio for the frame
    
    # show the frame
 #   cv2.imshow("Frame1", frame1)
 #   key = cv2.waitKey(1) & 0xFF
     
    # if the `q` key was pressed, break from the loop


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

        print('Start Recognition')
        prevTime = 0
        cnt=0
        while cnt<60:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
            count()
            curTime = time.time()+1    # calc fps
            timeF = frame_interval

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
                        print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        # print("predictions")
                        print(best_class_indices,' with accuracy ',best_class_probabilities)
                        cnt+=1
                        ar[best_class_indices[0]]+=1
                        # print(best_class_probabilities)
                        if best_class_probabilities>0.53:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            #print('Result Indices: ', best_class_indices[0])
                            print(HumanNames)
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    print(result_names)
                                    
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                            
                else:
                    print('Alignment Failure')
            # c+=1
            cv2.imshow('Video', frame)
            if cnt==60:
                print(ar)
                print("COUNTER : ",COUNTER)
                for po in range(0,listsize):
                    if m<ar[po]:
                        m=ar[po]
                        p=po
                print('recognition : ',HumanNames[p])
                if COUNTER==0:
                    print("NOT REAL")
                else:
                    print("REAL")
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                
                break

        video_capture.release()
        cv2.destroyAllWindows()
