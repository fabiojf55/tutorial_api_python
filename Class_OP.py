import sys
import cv2
import os
import argparse
import functions_general as fg

# Import Openpose
dir_path = os.path.dirname(os.path.realpath(__file__))

# Change these variables to point to the correct folder (Release/x64 etc.)
sys.path.append(dir_path + '/../../python/openpose/Release');

os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
try:
    import pyopenpose as op
except ImportError as e:
    print('Could not import OpenPose library. Compile again till it works')
    raise e

class Detector_OP():

    def __init__(self, path="Data/ME/", process_img = True, vid_label = "video.avi"):

        self.process_img = process_img
        self.vid_label = vid_label
        self.path = path
        # Flags
        parser = argparse.ArgumentParser()

        if self.process_img:
            parser.add_argument("--image_dir", default=path,
                                help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
            parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
        else:
            parser.add_argument("--video_path", default= (self.path + "/" + self.vid_label), help="Process a video. Read all standard formats (mp4, etc.).")

        self.args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["number_people_max"] = 1
        params["model_folder"] = "../../../models/"

        # Add others in path?
        if self.process_img:
            for i in range(0, len(self.args[1])):
                curr_item = self.args[1][i]
                if i != len(self.args[1]) - 1:
                    next_item = self.args[1][i + 1]
                else:
                    next_item = "1"
                if "--" in curr_item and "--" in next_item:
                    key = curr_item.replace('-', '')
                    if key not in params:  params[key] = "1"
                elif "--" in curr_item and "--" not in next_item:
                    key = curr_item.replace('-', '')
                    if key not in params: params[key] = next_item


        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def findKeypoints(self, img):
        datum = op.Datum()
        # img = fg.scale_img(imageToProcess, 0.33, 0, 0)
        datum.cvInputData = img
        self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        joints = datum.poseKeypoints  # Keypoints of OpenPose in a tuple

        # cv2.imshow("OpenPose", datum.cvOutputData)

        # "datum.poseKeypoints" gibt entweder ein Tuple mit [][][] zurück oder None deswegen folgender shit
        try:
            a = joints.size
            go = True
        except AttributeError:
            go = False

        keypoints_op = []
        if go:
            for i in range(int(joints.size / 3)):  # Turning keypoints of OpenPose in a list
                    x = int(joints[0][i][0])
                    y = int(joints[0][i][1])
                    keypoints_op.append([i, x, y])

        return keypoints_op

def main():

    detector = Detector_OP()
    # Read frames on directory
    imagePaths = op.get_images_on_directory(detector.args[0].image_dir);

    # Process and display images
    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        kp = detector.findKeypoints(img)
        print(kp)

if __name__ == "__main__":
    main()