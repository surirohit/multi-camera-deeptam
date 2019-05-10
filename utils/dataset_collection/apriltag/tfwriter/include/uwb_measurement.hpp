// TODO: create info to this file

#pragma once

#include "typedefs.hpp"
#include <numeric>
#include <ros/ros.h>
#include <string>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include "tf_writer/StampedDistance.h"
#include <math.h>
#include <map>
#include <random>
#include <cmath>
#include <iomanip>


using std::vector;
using std::string;

namespace cfe {

class UwbMeasurement {
public:

public:
    UwbMeasurement(ros::NodeHandle& nh);

private:
    geometry_msgs::PoseStamped msg_pose_mav1[1000],msg_pose_mav2[1000],msg_pose_mav3[1000],msg_pose_mav4[1000],msg_pose_mav5[1000];
    geometry_msgs::PoseStamped msg_pose_temp;
    bool time_sync_flag=false;
    short int timediff_threshold;
    short int time_sync_counter;
    point3d p_O1, p_O2, p_O3, p_O4, p_O5;
    double distance_O1_O2, distance_O1_O3, distance_O1_O4, distance_O1_O5;
    double distance_O2_O3, distance_O2_O4, distance_O2_O5;
    double distance_O3_O4, distance_O3_O5;
    double distance_O4_05;
    int queue_size_measurement;
    short int gt_topic_flags[5] ={0,0,0,0,0};
    short int gt_topics_pointer[5]={0,0,0,0,0};
    short int buffer_minimum;
    bool call_DistanceMeasurement;
    float uwb_std;

    centralized_formation_estimator::StampedDistance msg_uwb;







private:
    ros::NodeHandle nodeHandle;



    bool check_sync_data();
    void uwb_calculation();
    double abs_distance(point3d A, point3d B);


    ros::Subscriber mav1_gt_position_subs;
    ros::Subscriber mav2_gt_position_subs;
    ros::Subscriber mav3_gt_position_subs;
    ros::Subscriber mav4_gt_position_subs;
    ros::Subscriber mav5_gt_position_subs;

    ros::Publisher distance_O1_O2_publ, distance_O1_O3_publ, distance_O1_O4_publ, distance_O1_O5_publ;
    ros::Publisher distance_O2_O3_publ, distance_O2_O4_publ, distance_O2_O5_publ;
    ros::Publisher distance_O3_O4_publ, distance_O3_O5_publ;
    ros::Publisher distance_O4_O5_publ;

    void mav1_gt_position_Callback(const geometry_msgs::PoseStamped::ConstPtr& pose_gt_msgs);
    void mav2_gt_position_Callback(const geometry_msgs::PoseStamped::ConstPtr& pose_gt_msgs);
    void mav3_gt_position_Callback(const geometry_msgs::PoseStamped::ConstPtr& pose_gt_msgs);
    void mav4_gt_position_Callback(const geometry_msgs::PoseStamped::ConstPtr& pose_gt_msgs);
    void mav5_gt_position_Callback(const geometry_msgs::PoseStamped::ConstPtr& pose_gt_msgs);






}; //class uwbmeasurement





} // namespace cfe
