# import the necessary packages
from numpy.lib.type_check import imag
from scipy.spatial import distance as dist
import numpy as np
import cv2
from math import sqrt
# import imutils

def compute_point_perspective_transformation(matrix,boxes):
    list_downoids = [[box[4], box[5]+(box[3]//2)] for box in boxes]
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    transformed_points_list = list()
    for i in range(0,transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
    return np.array(transformed_points_list).astype('int')

def eucledian_distance(point1, point2):
    x1,y1 = point1
    x2,y2 = point2
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def get_red_green_boxes(distance_allowed,birds_eye_points,boxes):
    red_boxes = []
    green_boxes = []

    new_boxes = [tuple(box) + tuple(result) for box, result in zip(boxes, birds_eye_points)]
    for i in range(0, len(new_boxes)-1):
            for j in range(i+1, len(new_boxes)):
                cxi,cyi = new_boxes[i][6:]
                cxj,cyj = new_boxes[j][6:]
                distance = eucledian_distance([cxi,cyi], [cxj,cyj])
                if distance < distance_allowed:
                    red_boxes.append(new_boxes[i])
                    red_boxes.append(new_boxes[j])

    green_boxes = list(set(new_boxes) - set(red_boxes))
    red_boxes = list(set(red_boxes))
    return (green_boxes, red_boxes)

# source_points = np.array([(796, 180), (1518, 282), (1080, 719), (128, 480)], dtype="float32")

# image = cv2.imread("image.jpg")
# #image = imutils.resize(image, width=960)
# original_image_RGB_copy = image.copy()

# for point in source_points:
#     cv2.circle(original_image_RGB_copy, tuple(map(int, point)), 8, (255, 0, 0), -1)

# points = source_points.reshape((-1,1,2)).astype(np.int32)
# cv2.polylines(original_image_RGB_copy, [points], True, (0,255,0), thickness=4)

# dst=np.float32([(0.1,0.5), (0.69, 0.5), (0.69,0.8), (0.1,0.8)])
# dst_size=(800,1080)
# dst = dst * np.float32(dst_size)

# H_matrix = cv2.getPerspectiveTransform(source_points, dst)

# warped = cv2.warpPerspective(original_image_RGB_copy, H_matrix, dst_size)

# cv2.imshow("Image", imutils.resize(original_image_RGB_copy, width=960))
# cv2.imshow("Image", imutils.resize(warped, height=700))
# cv2.waitKey(0)
# cv2.destroyAllWindows()