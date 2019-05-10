// TODO: create info to this file

#include "ros/ros.h"
#include "std_msgs/String.h"
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <Eigen/StdVector>
#include <Eigen/StdDeque>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{
  //Eigen Variable
  Eigen::Translation<double,3> pos_cam_tag;
  Eigen::Quaterniond rot_cam_tag;
  Eigen::Transform<double,3,Eigen::Affine> pose_cam_tag;
  long int timecounter=0;

  ros::init(argc, argv, "tfwriter_node");
  ros::NodeHandle nh;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener(tfBuffer);
  ros::Rate rate(10.0);
  std::ofstream myfile;

  myfile.open ("groundtruth.txt");
  myfile << "#  Timestamps tx, ty, tz, qx, qy, qz, qw\n";

  while (nh.ok()) {
      geometry_msgs::TransformStamped tfstamped;
      try {
          //get tf camera to tag 15
          tfstamped = tfBuffer.lookupTransform("camera",
          "tag_15", ros::Time(0));

          pos_cam_tag.x()=tfstamped.transform.translation.x;
          pos_cam_tag.y()=tfstamped.transform.translation.y;
          pos_cam_tag.z()=tfstamped.transform.translation.z;
          rot_cam_tag.w() = tfstamped.transform.rotation.w;
          rot_cam_tag.vec() << tfstamped.transform.rotation.x,
                  tfstamped.transform.rotation.y,
                  tfstamped.transform.rotation.z;

//          ROS_INFO_STREAM(pos_cam_tag.x()<<" "<<pos_cam_tag.y());
//          ROS_INFO_STREAM(tfstamped.transform.translation.x<<" "<<tfstamped.transform.translation.y);

          //Get 4x4 transform
          pose_cam_tag = pos_cam_tag * rot_cam_tag;
          //Inverse to get tag to camera
          pose_cam_tag = pose_cam_tag.inverse();
          Eigen::Quaterniond rot_tag_cam(pose_cam_tag.rotation());


          myfile << timecounter << " "
                 << pose_cam_tag.translation()[0] << " " << pose_cam_tag.translation()[1] << " " << pose_cam_tag.translation()[2] << " "
                 << rot_tag_cam.x() << " " << rot_tag_cam.y() << " " << rot_tag_cam.z() << " " << rot_tag_cam.w()
                 << "\n";

          timecounter++;



          }
      catch (tf2::TransformException &exception) {
          ROS_WARN("%s", exception.what());
          ros::Duration(1.0).sleep();
          continue;
      }
      rate.sleep();
  }
  myfile.close();
  return 0;
}
