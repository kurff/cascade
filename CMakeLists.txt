CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()
PROJECT(split)


add_definitions(-DUSE_OPENCV)
add_compile_options(-std=c++11)

include_directories(.)
include_directories(/usr/include/graphviz)
include_directories(/usr/local/cuda/include/)


include_directories(caffe-cascade/distribute/include/)
link_directories(../caffe-cascade/distribute/lib/ /usr/lib/x86_64-linux-gnu/)


add_custom_target(mytargetname ALL COMMAND ${CMAKE_COMMAND} -E make_directory ./test/)

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d)
aux_source_directory(./test TEST_LIST)
aux_source_directory(./tools CONVERT_SRC)
set(BOOST_LIBS boost_system boost_filesystem)
add_library(convert ${CONVERT_SRC} )
target_link_libraries(convert ${OPENCV_LIBS} ${BOOST_LIBS} caffe glog)
foreach(source ${TEST_LIST})
    add_executable(${source}.bin ${source} )
    target_link_libraries(${source}.bin convert glog gflags)
endforeach(source)


add_executable(test_train random-forest/test_train.cpp)
target_link_libraries(test_train glog opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d gvc cgraph)

add_executable(test_deploy examples/test_deploy.cpp )
target_link_libraries(test_deploy caffe glog opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_features2d boost_system)