
GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver.prototxt >&1 | tee log.txt