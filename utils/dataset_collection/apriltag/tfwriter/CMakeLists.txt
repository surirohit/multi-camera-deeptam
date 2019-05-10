cmake_minimum_required(VERSION 2.8.3)
project(tf_writer)

## Use C++11
add_definitions(-std=c++11)


set(CMAKE_BUILD_TYPE RelWithDebInfo)

## Find catkin macros and libraries
find_package(catkin_simple REQUIRED)
#    catkin_simple
#    geometry_msgs
#    roscpp
#    sensor_msgs
#    message_generation
#    std_msgs
#    genmsg
#    )

# Add message files
catkin_simple(ALL_DEPS_REQUIRED)

#add_message_files(DIRECTORY msg
#  FILES
#  StampedDistance.msg
#)

#generate_messages()
#  DEPENDENCIES
#  std_msgs
#)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
#catkin_package(
#  INCLUDE_DIRS
#    include
##    ${EIGEN3_INCLUDE_DIR}
#  LIBRARIES
#    ${PROJECT_NAME}_core
#  CATKIN_DEPENDS
##    catkin
#    roscpp
#    sensor_msgs
#    message_runtime
##  CFG_EXTRAS catkin_simple-extras.cmake
#  DEPENDS
#    Eigen
#    Boost
#)



###########
## Build ##
###########
## Specify additional locations of header files
## Your package locations should be listed before other locations
#include_directories(
#  include
#  ${catkin_INCLUDE_DIRS}
#)





## Summarize the source files
### Declare a C++ executable
##centralized_formation_estimator_node
#add_executable(${PROJECT_NAME}
#  src/centralized_formation_estimator_node.cpp
#  src/state_filter.cpp
#  )

### Specify libraries to link a library or executable target against

##cs_add_library(${PROJECT_NAME} ${CENTRALIZED_FORMATION_ESTIMATOR_SOURCE})
#target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
## Observation Node
##set(UWB_MEASUREMENT_SOURCE
##    src/distance_generator.cpp
##    )
#add_executable(${PROJECT_NAME}_distance
#  src/distance_generator.cpp
#  src/distance_generator_node.cpp
#  )
#target_link_libraries(${PROJECT_NAME}_distance ${catkin_LIBRARIES})
#### Declare a C++ executable
# Summarize the source files
set(CENTRALIZED_FORMATION_ESTIMATOR_SOURCE
#  src/centralized_formation_estimator_node.cpp
  src/tfwriter.cpp
#  src/distance_generator_node.cpp
)
cs_add_library(${PROJECT_NAME} ${CENTRALIZED_FORMATION_ESTIMATOR_SOURCE})

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})

cs_add_executable(tfwriter_node
  src/tfwriter.cpp)
target_link_libraries(tfwriter_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES})

cs_install()
cs_export()
