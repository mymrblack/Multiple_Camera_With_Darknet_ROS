from __future__ import print_function
import rospy
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes, CorBBox
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sort import Sort
#from sort_origin import Sort

tracker = Sort()

def callback(data):
    """
    :param data: CorBBox type
           data.bounding_boxes type : list
    """
    rospy.loginfo("I have got data !")
    # print(type(data))
    # print(type(data.bounding_boxes))
    # print(type(data.bounding_boxes[0]))
    # print(help(data.bounding_boxes))
    
    ## get detections: 
    bbox_num = len(data.bounding_boxes)
    detections = []
    for i, bbox in enumerate(data.bounding_boxes):
        detection = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, bbox.probability]
        detections.append(detection)
    detections = np.array(detections)
    # init tracker
    
    track_bbs_ids = tracker.update(detections)
    #
    print(type(track_bbs_ids))
    print('number of bbox: ' + str(len(track_bbs_ids)))
    if len(track_bbs_ids) != 0:
        print(track_bbs_ids[0])

    #
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(data.image, "bgr8")
    except CvBridgeError as e:
        print(e)
    # draw text
    for track_bbs in track_bbs_ids:
        text_x = int(track_bbs[0])
        text_y = int(track_bbs[1]) - 30
        text = "id: " + str(int(track_bbs[-1]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        cv2.putText(cv_image, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, 8, 0)
    # show image 
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)


def listener():
    rospy.init_node('listener', anonymous=True)
    print('Hello')
    rospy.Subscriber('/darknet_ros/CorBBox_depth_image', CorBBox, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
