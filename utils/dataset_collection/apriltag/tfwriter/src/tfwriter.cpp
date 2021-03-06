// TODO: create info to this file

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Header.h"
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <Eigen/StdVector>
#include <Eigen/StdDeque>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "tf_writer/AprilTagDetection.h"
#include <iostream>
#include <fstream>

//void tagCallback(const tf_writer::AprilTagDetectionArray& msg) {
////    if(msg.detections[0==19){
//        ROS_INFO("Yes cam to 19");
////    }
//}

int main(int argc, char **argv)
{
  //Eigen Variable
  Eigen::Translation<double,3> pos_cam_tag;
  Eigen::Quaterniond rot_cam_tag;
  Eigen::Transform<double,3,Eigen::Affine> pose_cam_tag;

//  tf_writer::AprilTagDetectionArray msg_tagpose;

  ros::init(argc, argv, "tfwriter_node");
  ros::NodeHandle nh;

//  tf2_ros::Buffer tfBuffer;
//  tf2_ros::TransformListener tfListener(tfBuffer);
//  ros::Rate rate(10.0);
  std::ofstream myfile;

  myfile.open ("groundtruth.txt");
  myfile << "#  Timestamps tx, ty, tz, qx, qy, qz, qw\n";
//  ros::Subscriber posesubs = nh.subscribe("tag_detections",10,tagCallback);

//  while (nh.ok()) {

////      geometry_msgs::TransformStamped tfstamped;
////      try {
////          //get tf camera to tag 15
////          tfstamped = tfBuffer.lookupTransform("camera",
////          "tag_19", ros::Time(0));

//          pos_cam_tag.x()=tfstamped.transform.translation.x;
//          pos_cam_tag.y()=tfstamped.transform.translation.y;
//          pos_cam_tag.z()=tfstamped.transform.translation.z;
//          rot_cam_tag.w() = tfstamped.transform.rotation.w;
//          rot_cam_tag.vec() << tfstamped.transform.rotation.x,
//                  tfstamped.transform.rotation.y,
//                  tfstamped.transform.rotation.z;
//          long int timestamp_sec = tfstamped.header.stamp.sec;
//          long int timestamp_nsec = tfstamped.header.stamp.nsec;

////          ROS_INFO_STREAM(tfstamped.transform.translation.x<<" "<<tfstamped.transform.translation.y<<" "<<tfstamped.transform.translation.z);

//          //Get 4x4 transform
//          pose_cam_tag = pos_cam_tag * rot_cam_tag;
//          //Inverse to get tag to camera
//          pose_cam_tag = pose_cam_tag.inverse();
//          Eigen::Quaterniond rot_tag_cam(pose_cam_tag.rotation());

//          myfile << timestamp_sec <<"" << timestamp_nsec << " "
//                 << pose_cam_tag.translation()[0] << " " << pose_cam_tag.translation()[1] << " " << pose_cam_tag.translation()[2] << " "
//                 << rot_tag_cam.x() << " " << rot_tag_cam.y() << " " << rot_tag_cam.z() << " " << rot_tag_cam.w()
//                 << "\n";

//          timecounter++;

//          }
//      catch (tf2::TransformException &exception) {
//          ROS_WARN("%s", exception.what());
//          ros::Duration(1.0).sleep();
//          continue;
//      }
//      rate.sleep();
//  }
  ros::spin();
  myfile.close();
  return 0;
}
