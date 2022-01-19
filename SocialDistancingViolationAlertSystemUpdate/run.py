# USAGE
# python run.py --input pedestrians.mp4
# python run.py --input pedestrians.mp4 --output output.avi

# import the necessary packages
from SocialDistancingViolationAlertSystem import config
from SocialDistancingViolationAlertSystem.detection import detect_people
from SocialDistancingViolationAlertSystem.bird_eye import compute_point_perspective_transformation, get_red_green_boxes
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# loop over the frames from the video stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	
	# make copies of the real frame for the purpose of displaying results on them
	original_image_RGB_copy = frame.copy()
	new_box_image = frame.copy()

	# resize the frame and then detect people (and only people) in it
	results, boxes = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

	# initialize the set of indexes that violate the minimum social
	# distance
	violate = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
	if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
				if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of
					# the centroid pairs
					violate.add(i)
					violate.add(j)

	# loop over the results
	for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)

		# if the index pair exists within the violation set, then
		# update the color
		if i in violate:
			color = (0, 0, 255)

		# draw (1) a bounding box around the person and (2) the
		# centroid coordinates of the person,
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)

	#Accumulate the frame height and width and store them in variables
	image_height, image_width = frame.shape[:2]

	# Manually declaring the parameters required for bird-eye-view transformation
	dst=np.float32([(0.1,0.5), (0.69, 0.5), (0.69,0.8), (0.1,0.8)])
	dst_size=(800,1080)
	dst = dst * np.float32(dst_size)
	source_points = np.array([(796, 180), (1518, 282), (1080, 719), (128, 480)], dtype="float32")

	# Accumulate and Store transformation matrix
	H_matrix = cv2.getPerspectiveTransform(source_points, dst)

	# Mark the manually selected points on the frame copy
	for point in source_points:
		cv2.circle(original_image_RGB_copy, tuple(map(int, point)), 8, (255, 0, 0), -1)

	# Draw polylines to show the ROI selected by manual points and display the warped video
	points = source_points.reshape((-1,1,2)).astype(np.int32)
	cv2.polylines(original_image_RGB_copy, [points], True, (0,255,0), thickness=4)
	warped = cv2.warpPerspective(original_image_RGB_copy, H_matrix, dst_size)
	cv2.imshow("Warped Video", imutils.resize(warped, width=480))

	# Tranform the centroids detected to the warped perspective
	birds_eye_points = compute_point_perspective_transformation(H_matrix, boxes)

	# Extract the green boxes and red boxes for bird-eye-view
	green_box, red_box = get_red_green_boxes(config.MIN_DISTANCE, birds_eye_points, boxes)

	# Read a template image to display the results
	blank_image = cv2.imread("black_background.png")

	# Editing the image for displaying bird-eye-view persepective of the detections in the video
	cv2.putText(blank_image, str(len(red_box)), (120,100), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,0,255), 4, cv2.LINE_AA) 
	cv2.putText(blank_image, str(len(green_box)), (520,100), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 4, cv2.LINE_AA)
    
	for point in green_box:
         cv2.circle(blank_image,tuple([point[6],point[7]]),20,(0,255,0),-1)
	for point in red_box:
		cv2.circle(blank_image,tuple([point[6],point[7]]),20,(0,0,255),-1)
	eye_view_height=405
	eye_view_width=360
	blank_image = cv2.resize(blank_image,(eye_view_width,eye_view_height))
	cv2.imshow("Bird Eye View", blank_image)

	# Displaying the detections as per the bird-eye-view
	for point in green_box:
		cv2.rectangle(new_box_image,(point[0],point[1]),(point[0]+point[2],point[1]+point[3]),(0, 255, 0), 2)
	for point in red_box:
		cv2.rectangle(new_box_image,(point[0],point[1]),(point[0]+point[2],point[1]+point[3]),(0, 0, 255), 2)
	text = "Social Distancing Violations: {}".format(len(red_box))
	cv2.putText(new_box_image, text, (10, new_box_image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
	cv2.imshow("From Bird Eye View", imutils.resize(new_box_image, width=720))


	# draw the total number of social distancing violations on the
	# output frame
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", imutils.resize(frame, width=720))
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)
