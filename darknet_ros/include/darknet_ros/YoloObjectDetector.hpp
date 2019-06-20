/*
 * YoloObjectDetector.h
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#pragma once

// c++
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
#include <pthread.h>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>

// ROS
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Int8.h>
#include <actionlib/server/simple_action_server.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>

// OpenCv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>

// darknet_ros_msgs
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>

#include <darknet_ros_msgs/CorBBox.h>
// Darknet.
#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif

extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "darknet_ros/image_interface.h"
#include <sys/time.h>
}

extern "C" void ipl_into_image(IplImage* src, image im);
extern "C" image ipl_to_image(IplImage* src);
extern "C" void show_image_cv(image p, const char *name, IplImage *disp);
extern "C" void rgbgr_image(image im);


//-------------------------Added By Lynne Begin -----------------
#define CAMERA_NUM 2 //defined by lynne.
#define SUBCRIBE_CAM_TOPIC_PATH_ROOT "subscribers/camera_reading_"
#define CAMERA_TOPIC_DEFAULT_NAME "/camera/image_raw_"
#define PUBLISH_OBJECT_DETECTOR_PATH_ROOT "publishers/object_detector_"
#define OBJECT_DETECTOR_DEFAULT_NAME "found_object_"
#define PUBLISH_BOUNDING_BOXES_PATH_ROOT "publishers/bounding_boxes_"
#define BOUNDIGN_BOXES_DEFAULT_NAME "bounding_boxes_"
#define PUBLISH_DETECTION_IMAGE_PATH_ROOT "publishers/detection_image_"
#define DETECTION_IMAGE_DEFAULT_NAME "detection_image_"
#define THRESHOLD_PATH "yolo_model/threshold/value"
#define WEIGHTS_MODEL_PATH  "yolo_model/weight_file/name"
#define DEFAULT_WEIGHT_FILE_NAME "yolov3-tiny.weights"
#define CAMERA_DEPTH_TOPIC_PATH_ROOT "subscribers/camera_depth_reading_"
#define CAMERA_DEPTH_TOPIC_DEFAULT_NAME "/camera/depth/image_rect_raw_"


namespace darknet_ros {

//! Bounding box of the detected object.
typedef struct
{
  float x, y, w, h, prob;
  int num, Class;
} RosBox_;


// camera D415 intrincics
const double x0 = 315.297; // principle point
const double y0 = 248.041;
const double fx = 615.094; // focal length
const double fy = 614.703;

typedef struct
{
  IplImage* image;
  std_msgs::Header header;
} IplImageWithHeader_;

typedef struct{

  std::string cameraTopicNames[CAMERA_NUM]; //Added by Lynne.
  int cameraQueueSize;

  std::string objectDetectorTopicNames[CAMERA_NUM];
  int objectDetectorQueueSize;
  bool objectDetectorLatch;

  std::string boundingBoxesTopicNames[CAMERA_NUM];
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;

  std::string detectionImageTopicNames[CAMERA_NUM];
  int detectionImageQueueSize;
  bool detectionImageLatch;

  std::string depthTopicNames[CAMERA_NUM];
  int depthQueueSize;

  std::string corbboxTopicName;
  int corbboxQueueSize;
  bool corbboxLatch;

  std::string checkForObjectsActionName;
} TopicInputParams_;

class YoloObjectDetector
{
 public:
  /*!
   * Constructor.
   */
  explicit YoloObjectDetector(ros::NodeHandle nh);

  /*!
   * Destructor.
   */
  ~YoloObjectDetector();

 private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  TopicInputParams_ topicParameters;
  void readTopicParameters();
  /*!
   * Initialize the ROS connections.
   */
  void init();

  /*!
   * Callback of camera.
   * @param[in] msg image pointer.
   */
  void cameraCallback_1(const sensor_msgs::ImageConstPtr& msg);
  void cameraCallback_2(const sensor_msgs::ImageConstPtr& msg);


  //! callback function for subscriber of depth image
  void depth_image_Callback(const sensor_msgs::ImageConstPtr& msg);

  /*!
   * Check for objects action goal callback.
   */
  void checkForObjectsActionGoalCB();

  /*!
   * Check for objects action preempt callback.
   */
  void checkForObjectsActionPreemptCB();

  /*!
   * Check if a preempt for the check for objects action has been requested.
   * @return false if preempt has been requested or inactive.
   */
  bool isCheckingForObjects() const;

  /*!
   * Publishes the detection image.
   * @return true if successful.
   */
  bool publishDetectionImage(const cv::Mat& detectionImage);

  //! Typedefs.
  typedef actionlib::SimpleActionServer<darknet_ros_msgs::CheckForObjectsAction> CheckForObjectsActionServer;
  typedef std::shared_ptr<CheckForObjectsActionServer> CheckForObjectsActionServerPtr;

  //! ROS node handle.
  ros::NodeHandle nodeHandle_;

  //! Class labels.
  int numClasses_;
  std::vector<std::string> classLabels_;

  //! Check for objects action server.
  CheckForObjectsActionServerPtr checkForObjectsActionServer_;

  //! Advertise and subscribe to image topics.
  image_transport::ImageTransport imageTransport_;

  //! ROS subscriber and publisher.
  image_transport::Subscriber imageSubscriber_[CAMERA_NUM];

  //! Detected objects.
  std::vector<std::vector<RosBox_> > rosBoxes_;
  std::vector<int> rosBoxCounter_;
  //darknet_ros_msgs::BoundingBoxes boundingBoxesResults_[CAMERA_NUM];
  darknet_ros_msgs::BoundingBoxes boundingBoxesResults_;

  //! Camera related parameters.
  int frameWidth_;
  int frameHeight_;


  //SortObject sortobject;
  cv::Mat depth_image; // store depth image matrix
  //! Subscriber of the depth image
  ros::Subscriber depthImageSubscriber_[CAMERA_NUM];
  //! Publisher of the bounding box with (x,y,z)
  ros::Publisher corDepthImagePublisher_;

  ros::Publisher objectPublisher_[CAMERA_NUM];
  ros::Publisher boundingBoxesPublisher_[CAMERA_NUM];

  //! Publisher of the bounding box image.
  ros::Publisher detectionImagePublisher_[CAMERA_NUM];


  // Yolo running on thread.
  std::thread yoloThread_;

  // Darknet.
  char **demoNames_;
  image **demoAlphabet_;
  int demoClasses_;

  network *net_;
  std_msgs::Header headerBuff_[3];
  //std_msgs::Header linkedHeaderBuff_[3];
  image buff_[3];
  image buffLetter_[3];
  int buffId_[3];
  int buffIndex_ = 0;
  IplImage * ipl_;
  float fps_ = 0;
  float demoThresh_ = 0;
  float demoHier_ = .5;
  int running_ = 0;

  int demoDelay_ = 0;
  int demoFrame_ = 3;
  float **predictions_;
  int demoIndex_ = 0;
  int demoDone_ = 0;
  float *lastAvg2_;
  float *lastAvg_;
  float *avg_;
  int demoTotal_ = 0;
  double demoTime_;

  RosBox_ *roiBoxes_;
  bool viewImage_;
  bool enableConsoleOutput_;
  int waitKeyDelay_;
  int fullScreen_;
  char *demoPrefix_;

  std_msgs::Header imageHeader_;
  cv::Mat camImageCopy_;
  boost::shared_mutex mutexImageCallback_;

  bool imageStatus_ = false;
  boost::shared_mutex mutexImageStatus_;

  bool isNodeRunning_ = true;
  boost::shared_mutex mutexNodeStatus_;

  int actionId_;
  boost::shared_mutex mutexActionStatus_;

  // double getWallTime();

  int sizeNetwork(network *net);

  void rememberNetwork(network *net);

  detection *avgPredictions(network *net, int *nboxes);

  void *detectInThread();

  void *fetchInThread();

  void *displayInThread(void *ptr);

  void *displayLoop(void *ptr);

  void *detectLoop(void *ptr);

  void setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                    char **names, int classes,
                    int delay, char *prefix, int avg_frames, float hier, int w, int h,
                    int frames, int fullscreen);

  void yolo();

  IplImageWithHeader_ getIplImageWithHeader();

  bool getImageStatus(void);

  bool isNodeRunning(void);

  void *publishInThread();

//get camera topic names. Add by Lynne.
//void getCameraTopic(std::string cameraTopicNames[]);
  
  void getTopicNameParameters(const std::string pathRoot, std::string topicNames[], int cam_num, const std::string defaultName);
  
  void publishTopics();

  void subscribeTopics();

  void setupActionServer();

  void waitingForNewImage();
  
  bool newImageComeFlag_ = false;
  std::mutex mutexNewImageCome;
  std::condition_variable newImageComeCondition;

  std::string depthTopicName;
  int depthQueueSize;

  std::string corbboxTopicName;
  int corbboxQueueSize;
  bool corbboxLatch;

  void imagePreprocess(image p, IplImage *disp);
};

} /* namespace darknet_ros*/
