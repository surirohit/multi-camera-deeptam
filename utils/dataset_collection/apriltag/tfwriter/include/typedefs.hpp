// TODO: create info to this file

#pragma once

#include <memory>
#include <utility>

#include <Eigen/StdVector>
#include <Eigen/StdDeque>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
//#include <Eigen/>

namespace cfe {

// Typedefs for Eigen types
typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> >
  PointVector;
typedef std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> >
  QuaternionVector;
typedef Eigen::Quaterniond Quaternion;
typedef Eigen::Transform<double,3,Eigen::Affine> PoseTransform;
typedef Eigen::Translation<double,3> PoseTranslation;
typedef Eigen::Matrix<double, 28, 1> StateQuat5;
//typedef std::vector<Eigen::Transform, Eigen::aligned_allocator<Eigen::Transform> > PoseTransformVector;

// Typedefs


typedef struct Point{
    double x,y,z;
} point3d;
//ADD Transformation Matrices
//ADD States
//ADD ....
//  typedef std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f>> VectorAffine3f;


} // namespace cfe

