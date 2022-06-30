import sys
import cv2
import os
from sys import platform
import argparse
try:
	import pyopenpose as op
except ImportError as e:
    print('Could not import OpenPose library. Compile again till it works')
    raise e

#Setting OpenPose parameters
def set_params():

	params = dict()
	#params["logging_level"] = 3
	#params["output_resolution"] = "-1x-1"
	#params["net_resolution"] = "-1x368"
	#params["model_pose"] = "BODY_25"
	#params["alpha_pose"] = 0.6
	#params["scale_gap"] = 0.3
	#params["scale_number"] = 1
	#params["render_threshold"] = 0.05
	# If GPU version is built, and multiple GPUs are available, set the ID he
	#params["number_gpu"] = 1
	#params["disable_blending"] = False
	# Ensure you point to the correct path where models are located
	params["number_people_max"] = 1
	params["model_folder"] = "../../models/"
	return params

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--video_path", default="loco_vid_1.mp4", help="Process a video. Read all standard formats (mp4, etc.).")
	args = parser.parse_known_args()

	params = set_params()

	#Constructing OpenPose object allocates GPU memory
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()

	# Process Video
	datum = op.Datum()
	cap = cv2.VideoCapture(args[0].video_path)
	# cap = cv2.VideoCapture(1)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	output_name = str(args[0].video_path)[:-4] + "_output.avi"
	out = cv2.VideoWriter(output_name ,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

	while cap.isOpened():
		ret,frame=cap.read()
		if ret == True:
			image = frame #cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
			datum.cvInputData = image
			opWrapper.emplaceAndPop([datum])

			# Output keypoints and the image with the human skeleton blended on it
			output_image = datum.cvOutputData

			# Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
			#print('Pose estimated!')

               		# Save the stream
	                # cv2.putText(output_image,'OpenPose using Python-OpenCV',(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)
			out.write(output_image)

		else:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()
