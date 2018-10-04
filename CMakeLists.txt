CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(split)
add_compile_options(-std=c++11)

include_directories(.)

add_executable(split tools/split_image.cpp)
target_link_libraries(split opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d)
add_executable(test_train random-forest/test_train.cpp)
target_link_libraries(test_train glog opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d)