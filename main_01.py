import cv2
import os
from Class_MP import Detector_MP
from Class_OP import Detector_OP
import functions_general as fg

process_img = False                                                     # Want to process a video (False) or images (True)?

if process_img:
    path = "Data/ME/"                                                   # Path where images are saved
    image_names = os.listdir(path)                                      # List of all images in given path
else:
    path = "Data/"
    video_name = "Test_01.mp4"

detector_MP = Detector_MP()                                             # Create instance of Mediapipe pose Detector
detector_OP = Detector_OP(path=path, process_img=process_img)           # Create instance of OpenPose pose Detector


# Process and display images
if process_img:
    for image_name in image_names:
        imagePath = os.path.join(path, image_name)                      # Append current image to path
        imageToProcess = cv2.imread(imagePath)
        img = fg.scale_img(imageToProcess, 0.33, 0, 0)                  # Scale image if needed (img, scalefactor, width, height)

        keypoints_op = detector_OP.findKeypoints(img)                   # Keypoints of OpenPose
        keypoints_mp = detector_MP.findKeypoints(img, draw=False)       # Keypoints of Mediapipe, if second input True --> Draw skeleton of MP

        fg.draw_keypoints(img, keypoints_mp, keypoints_op)              # Visual comparison between both models

        print("Body keypoints Mediapipe: \n" + str(keypoints_mp))
        print("Body keypoints OpenPose: \n" + str(keypoints_op))

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
else:
    cap = cv2.VideoCapture(path+video_name)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            keypoints_op = detector_OP.findKeypoints(frame)
            keypoints_mp = detector_MP.findKeypoints(frame, draw=False)
            fg.draw_keypoints(frame, keypoints_mp, keypoints_op)        # Visual comparison between both models

            print("Body keypoints Mediapipe: \n" + str(keypoints_mp))
            print("Body keypoints OpenPose: \n" + str(keypoints_op))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

print("Successfully finished.")
