CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(split)
add_compile_options(-std=c++11)

include_directories(.)
include_directories(/usr/include/graphviz)
include_directories(/usr/local/cuda/include/)

include_directories(caffe-cascade/distribute/include/)
link_directories(../caffe-cascade/distribute/lib/ /usr/lib/x86_64-linux-gnu/)

add_custom_target(mytargetname ALL COMMAND ${CMAKE_COMMAND} -E make_directory ./test/)

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d)
aux_source_directory(./test TEST_LIST)

add_executable(${TEST_LIST}.bin ${TEST_LIST})
target_link_libraries(${TEST_LIST}.bin glog ${OPENCV_LIBS})



add_executable(split tools/split_image.cpp)
target_link_libraries(split opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d)
add_executable(test_train random-forest/test_train.cpp)
target_link_libraries(test_train glog opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d gvc cgraph)

add_executable(test_deploy examples/test_deploy.cpp )
target_link_libraries(test_deploy caffe glog opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d boost_system)