/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

char *cfg;
char *weights;
char *data;
char **detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_),
      numClasses_(0),
      classLabels_(0),
      rosBoxes_(0),
      rosBoxCounter_(0)
{
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector()
{
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
	{
	//The flag here just for continue the while loop in yolo() so that it can detect the isNodeRunning_
	  std::unique_lock<std::mutex> loc(mutexNewImageCome);
	  newImageComeFlag_ = true;
	  newImageComeCondition.notify_all();
	}
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters()
{
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_,
                    std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init()
{
  ROS_INFO("[YoloObjectDetector] init().");

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param( THRESHOLD_PATH, thresh, (float) 0.3);

  // Path to weights file.
  nodeHandle_.param(WEIGHTS_MODEL_PATH, weightsModel,
                    std::string(DEFAULT_WEIGHT_FILE_NAME));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char**) realloc((void*) detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_,
                0, 0, 1, 0.5, 0, 0, 0, 0);
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.

//--------------------- Added by Lynne Begin ------------------------------//

  TopicInputParams_ topicInputParams;
  getTopicInputParams(&topicInputParams);

//--------------------- Added by Lynne End ------------------------------//


  // subsrciber
  for (int i = 0; i < CAMERA_NUM; i++){
	  imageSubscriber_[i] = imageTransport_.subscribe( topicInputParams.cameraTopicNames[i], topicInputParams.cameraQueueSize, &YoloObjectDetector::cameraCallback, this);
	  depthImageSubscriber_[i]= nodeHandle_.subscribe(topicInputParams.depthTopicNames[i], topicInputParams.depthQueueSize, &YoloObjectDetector::depth_image_Callback, this);
  }
  /*
  imageSubscriber_[0] = imageTransport_.subscribe( topicInputParams.cameraTopicNames[0], topicInputParams.cameraQueueSize, &YoloObjectDetector::cameraCallback, this);
  imageSubscriber_[1] = imageTransport_.subscribe( topicInputParams.cameraTopicNames[1], topicInputParams.cameraQueueSize, &YoloObjectDetector::cameraCallback, this);
   depthImageSubscriber_[0]= nodeHandle_.subscribe(topicInputParams.depthTopicNames[0], topicInputParams.depthQueueSize, &YoloObjectDetector::depth_image_Callback, this);
   depthImageSubscriber_[1]= nodeHandle_.subscribe(topicInputParams.depthTopicNames[1], topicInputParams.depthQueueSize, &YoloObjectDetector::depth_image_Callback, this);
   */
//--------------------- Added by Lynne Begin ------------------------------//
  publishTopics(&topicInputParams);
//--------------------- Added by Lynne End ------------------------------//

  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName,
                    std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(
      new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(
      boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();
}


/*************************************** New Added *********************************************************/

void YoloObjectDetector::depth_image_Callback(const sensor_msgs::ImageConstPtr& msg)
{
	//ROS_INFO("depth_image_sub is running!");
	// convert message to opencv form
	cv_bridge::CvImagePtr cv_ptr; // cv image pointer
	try{
		cv_ptr = cv_bridge::toCvCopy(msg);
	}
	catch (cv_bridge::Exception& e){
        //if there is an error during conversion, display it
        ROS_ERROR("narutoNode.cpp::cv_bridge exception: %s", e.what());
        return;
    }
    // get depth image matrix
	//cv::Mat depth_image = cv_ptr->image;
	depth_image = cv_ptr->image;
}

/******************************************  End **************************************************************/
void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_DEBUG("[YoloObjectDetector] USB image received.");
  {
	  std::unique_lock<std::mutex> loc(mutexNewImageCome);
  	newImageComeFlag_ = true;
	newImageComeCondition.notify_all();
  }
  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = msg->header;
      //std::cout << imageHeader_ << std::endl;
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

/*
void YoloObjectDetector::getCameraTopic(std::string cameraTopicNames[]){
  std::string subscribeTopicPaths[CAMERA_NUM];
  std::string cameraNum;
  for(int i=1; i <= CAMERA_NUM; i++){
	cameraNum = std::to_string(i);	
	subscribeTopicPaths[i-1] = SUBCRIBE_CAM_TOPIC_PATH_ROOT + cameraNum + "/topic";
	std::cout << subscribeTopicPaths[i-1] << std::endl;
  	nodeHandle_.param(subscribeTopicPaths[i-1], cameraTopicNames[i-1],
                   std::string("/camera/image_raw"));
  }
}
*/

void YoloObjectDetector::getTopicNameParameters(const std::string pathRoot, std::string topicNames[], int cam_num, const std::string defaultName){

  std::string topicPaths[cam_num];
  std::string cameraNumString;

  for(int i=1; i <= cam_num; i++){
	cameraNumString = std::to_string(i);	
	topicPaths[i-1] = pathRoot + cameraNumString + "/topic";
	//std::cout << topicPaths[i-1] << std::endl; //for debug use.
  	nodeHandle_.param(topicPaths[i-1], topicNames[i-1],
                   defaultName + cameraNumString);
  }
}

void YoloObjectDetector::publishTopics(TopicInputParams_ *topicInputParams){

  objectPublisher_[0] = nodeHandle_.advertise<std_msgs::Int8>(topicInputParams->objectDetectorTopicNames[0],
                                                           topicInputParams->objectDetectorQueueSize,
                                                           topicInputParams->objectDetectorLatch);
  boundingBoxesPublisher_[0] = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(
      topicInputParams->boundingBoxesTopicNames[0], topicInputParams->boundingBoxesQueueSize, topicInputParams->boundingBoxesLatch);

  detectionImagePublisher_[0] = nodeHandle_.advertise<sensor_msgs::Image>(topicInputParams->detectionImageTopicNames[0],
                                                                       topicInputParams->detectionImageQueueSize,
                                                                       topicInputParams->detectionImageLatch);

   corDepthImagePublisher_= nodeHandle_.advertise<darknet_ros_msgs::CorBBox>(topicInputParams->corbboxTopicName, topicInputParams->corbboxQueueSize, topicInputParams->corbboxLatch);
}

void YoloObjectDetector::getTopicInputParams(TopicInputParams_ *topicInputParams){

  getTopicNameParameters(SUBCRIBE_CAM_TOPIC_PATH_ROOT, topicInputParams->cameraTopicNames, CAMERA_NUM, CAMERA_TOPIC_DEFAULT_NAME);

  getTopicNameParameters(PUBLISH_OBJECT_DETECTOR_PATH_ROOT, topicInputParams->objectDetectorTopicNames, CAMERA_NUM, OBJECT_DETECTOR_DEFAULT_NAME);

  getTopicNameParameters(PUBLISH_BOUNDING_BOXES_PATH_ROOT, topicInputParams->boundingBoxesTopicNames, CAMERA_NUM, BOUNDIGN_BOXES_DEFAULT_NAME);

  getTopicNameParameters(PUBLISH_DETECTION_IMAGE_PATH_ROOT, topicInputParams->detectionImageTopicNames, CAMERA_NUM, DETECTION_IMAGE_DEFAULT_NAME);

  getTopicNameParameters(CAMERA_DEPTH_TOPIC_PATH_ROOT, topicInputParams->depthTopicNames, CAMERA_NUM, CAMERA_DEPTH_TOPIC_DEFAULT_NAME);

  nodeHandle_.param("subscribers/camera_depth_reading_1/queue_size", topicInputParams->depthQueueSize, 1);
  nodeHandle_.param("subscribers/camera_reading_1/queue_size", topicInputParams->cameraQueueSize, 1);

  nodeHandle_.param("publishers/object_detector/queue_size", topicInputParams->objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", topicInputParams->objectDetectorLatch, false);

  nodeHandle_.param("publishers/bounding_boxes/queue_size", topicInputParams->boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", topicInputParams->boundingBoxesLatch, false);

  nodeHandle_.param("publishers/detection_image/queue_size", topicInputParams->detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", topicInputParams->detectionImageLatch, true);
  
  nodeHandle_.param("publishers/CorBBox_depth_image/topic", topicInputParams->corbboxTopicName, std::string("CorBBox_depth_image"));
  nodeHandle_.param("publishers/CorBBox_depth_image/queue_size", topicInputParams->corbboxQueueSize, 1);
  nodeHandle_.param("publishers/CorBBox_depth_image/latch", topicInputParams->corbboxLatch, true);
}

void YoloObjectDetector::checkForObjectsActionGoalCB()
{
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr =
      checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
      //std::cout << "actionID: " << actionId_ << std::endl;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionPreemptCB()
{
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const
{
  return (ros::ok() && checkForObjectsActionServer_->isActive()
      && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage)
{
  if (detectionImagePublisher_[0].getNumSubscribers() < 1 /*&& detectionImagePublisher_[1].getNumSubscribers() < 1 */)
    return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_[0].publish(*cvImage.toImageMsg());
  //detectionImagePublisher_[1].publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network *net)
{
  int i;
  int count = 0;
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection *YoloObjectDetector::avgPredictions(network *net, int *nboxes)
{
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for(j = 0; j < demoFrame_; ++j){
    axpy_cpu(demoTotal_, 1./demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for(i = 0; i < net->n; ++i){
    layer l = net->layers[i];
    if(l.type == YOLO || l.type == REGION || l.type == DETECTION){
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection *dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
  return dets;
}

void *YoloObjectDetector::detectInThread()
{
  running_ = 1;
  float nms = .4;

  layer l = net_->layers[net_->n - 1];
  float *X = buffLetter_[(buffIndex_ + 2) % 3].data;
  float *prediction = network_predict(net_, X);

  rememberNetwork(net_);
  detection *dets = 0;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes);

  if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

  if (enableConsoleOutput_) {
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps_);
    printf("Objects:\n\n");
  }
  image display = buff_[(buffIndex_+2) % 3];
  draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_);

  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
  for (i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0)
      xmin = 0;
    if (ymin < 0)
      ymin = 0;
    if (xmax > 1)
      xmax = 1;
    if (ymax > 1)
      ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j) {
      if (dets[i].prob[j]) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];
          count++;
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0) {
    roiBoxes_[0].num = 0;
  } else {
    roiBoxes_[0].num = count;
  }

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return 0;
}

void *YoloObjectDetector::fetchInThread()
{

  IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
  IplImage* ROS_img = imageAndHeader.image;
  ipl_into_image(ROS_img, buff_[buffIndex_]);

  headerBuff_[buffIndex_] = imageAndHeader.header;
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    buffId_[buffIndex_] = actionId_;
  }
  rgbgr_image(buff_[buffIndex_]);
  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  return 0;
}

void *YoloObjectDetector::displayInThread(void *ptr)
{
  if ( headerBuff_[(buffIndex_ + 1)%3].frame_id == "camera1_color_optical_frame"){
     show_image_cv(buff_[(buffIndex_ + 1)%3], "YOLO V3", ipl_);
  }
  else{
     show_image_cv(buff_[(buffIndex_ + 1)%3], "YOLO V3_1", ipl_);
  }
  int c = cvWaitKey(waitKeyDelay_);
  if (c != -1) c = c%256;
  if (c == 27) {
      demoDone_ = 1;
      return 0;
  } else if (c == 82) {
      demoThresh_ += .02;
  } else if (c == 84) {
      demoThresh_ -= .02;
      if(demoThresh_ <= .02) demoThresh_ = .02;
  } else if (c == 83) {
      demoHier_ += .02;
  } else if (c == 81) {
      demoHier_ -= .02;
      if(demoHier_ <= .0) demoHier_ = .0;
  }
  return 0;
}

void *YoloObjectDetector::displayLoop(void *ptr)
{
  while (1) {
    displayInThread(0);
  }
}

void *YoloObjectDetector::detectLoop(void *ptr)
{
  while (1) {
    detectInThread();
  }
}

void YoloObjectDetector::setupNetwork(char *cfgfile, char *weightfile, char *datafile, float thresh,
                                      char **names, int classes,
                                      int delay, char *prefix, int avg_frames, float hier, int w, int h,
                                      int frames, int fullscreen)
{
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image **alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V3\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo()
{
  const auto wait_duration = std::chrono::milliseconds(2000);

  while (!getImageStatus()) {
    printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;
  std::thread fetch_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float **) calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i){
      predictions_[i] = (float *) calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float *) calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_ *) calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
  IplImage* ROS_img = imageAndHeader.image;
  buff_[0] = ipl_to_image(ROS_img);
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  headerBuff_[0] = imageAndHeader.header;
  headerBuff_[1] = headerBuff_[0];
  headerBuff_[2] = headerBuff_[0];
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);

  int count = 0;

  if (!demoPrefix_ && viewImage_) {
    cvNamedWindow("YOLO V3", CV_WINDOW_NORMAL);
    cvNamedWindow("YOLO V3_1", CV_WINDOW_NORMAL);
    if (fullScreen_) {
      cvSetWindowProperty("YOLO V3", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
      cvSetWindowProperty("YOLO V3_1", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    } else {
      cvMoveWindow("YOLO V3", 0, 0);
      cvMoveWindow("YOLO V3_1", 0, 0);
      cvResizeWindow("YOLO V3", 640, 480);
      cvResizeWindow("YOLO V3_1", 640, 480);
    }
  }

  const auto wait_duration_1 = std::chrono::milliseconds(5);//Added By Lynne.
  demoTime_ = what_time_is_it_now();

  while (!demoDone_) {
//-------------Added By Lynne Begin------------------
	  std::unique_lock<std::mutex> loc(mutexNewImageCome);
	while(!newImageComeFlag_)
	{
		newImageComeCondition.wait(loc);
	}
	  /*
    if (!newImageComeFlag_) {
        if (!isNodeRunning()) {
          demoDone_ = true;
        }
    	std::this_thread::sleep_for(wait_duration_1);
	continue;	
    }
	*/

    newImageComeFlag_ = false;
//-------------Added By Lynne End------------------

    buffIndex_ = (buffIndex_ + 1) % 3;
    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
    if (!demoPrefix_) {
      fps_ = 1./(what_time_is_it_now() - demoTime_);
      demoTime_ = what_time_is_it_now();
      if (viewImage_) {
        displayInThread(0);
      }
      publishInThread();
    } else {
      char name[256];
      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(buff_[(buffIndex_ + 1) % 3], name);
    }
    fetch_thread.join();
    detect_thread.join();
    ++count;
    if (!isNodeRunning()) {
      demoDone_ = true;
    }
  }

}

IplImageWithHeader_ YoloObjectDetector::getIplImageWithHeader()
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
  IplImage* ROS_img = new IplImage(camImageCopy_);
  IplImageWithHeader_ header = {.image = ROS_img, .header = imageHeader_};
  return header;
}

bool YoloObjectDetector::getImageStatus(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void)
{
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void *YoloObjectDetector::publishInThread()
{
  /*************************************** New Added ********************************************************/

   darknet_ros_msgs::CorBBox depth_msg_to_publish;

/**********************************************************************************************************/
  // Publish image.
  if(!viewImage_){
      imagePreprocess(buff_[(buffIndex_ + 1)%3], ipl_);
   }
  
  cv::Mat cvImage = cv::cvarrToMat(ipl_);
  if (!publishDetectionImage(cv::Mat(cvImage))) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num > 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    std_msgs::Int8 msg;
    msg.data = num;
    objectPublisher_[0].publish(msg);
    //objectPublisher_[1].publish(msg);

    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        darknet_ros_msgs::BoundingBox boundingBox;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox.Class = classLabels_[i];
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);

/*************************************** New Added ********************************************************/

// get center point of box
        geometry_msgs::Point tempPoint;
// center point (xmin, ymin) --> pointcloud (x, y, z) unit:meters
	tempPoint.z = depth_image.at<unsigned short>(320, 240) / 1000.0;
	double xc = (xmin+xmax) / 2;
	double yc = (ymin+ymax) / 2;
	tempPoint.x = (xc - x0) * tempPoint.z / fx;
	tempPoint.y = (yc - y0) * tempPoint.z / fy;
        depth_msg_to_publish.coors.push_back(tempPoint);
	//cor_bbox_results_.cors.push_back(tempCor);
	//cor_bbox_results_.bbox.push_back(boundingBox);

/**********************************************************************************************************/
        }
      }
    }
    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = "detection";
    boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
    boundingBoxesPublisher_[0].publish(boundingBoxesResults_);
    //boundingBoxesPublisher_[1].publish(boundingBoxesResults_);
  } else {
    std_msgs::Int8 msg;
    msg.data = 0;
    objectPublisher_[0].publish(msg);
    //objectPublisher_[1].publish(msg);
  }

  /*************************************** New Added ********************************************************/

// Publish self-defined message
    depth_msg_to_publish.bounding_boxes = boundingBoxesResults_.bounding_boxes;
    depth_msg_to_publish.header.stamp = ros::Time::now();
    depth_msg_to_publish.header.frame_id = "coors_of_depth";
    sensor_msgs::ImagePtr im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cvImage).toImageMsg();
    depth_msg_to_publish.image = *im_msg;
// Publish self-defined message
    corDepthImagePublisher_.publish(depth_msg_to_publish);
/**********************************************************************************************************/
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();
  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return 0;
}

void YoloObjectDetector::imagePreprocess(image p, IplImage *disp)
{
    int x,y,k;

    if(p.c == 3) {
        rgbgr_image(p);
    }
 /*   if(p.c == 3){
        int i;
        for(i = 0; i < p.w*p.h; ++i){
       		 float swap = p.data[i];
       		 p.data[i] = p.data[i+p.w*p.h*2];
       		 p.data[i+p.w*p.h*2] = swap;
        }
    }
*/
    int step = disp->widthStep;

    for(y = 0; y < p.h; ++y){
        for(x = 0; x < p.w; ++x){
            for(k= 0; k < p.c; ++k){
                disp->imageData[y*step + x*p.c + k] = (unsigned char)((p.data[k*p.h*p.w + y*p.w + x])*255);
                //disp->imageData[y*step + x*p.c + k] = (unsigned char)(get_pixel(p,x,y,k)*255);
            }
        }
    }
}

} /* namespace darknet_ros*/
