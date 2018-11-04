
GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver.prototxt \
  --weights=../../../models/VGG_ILSVRC_16_layers_fc_reduced.caffemodel>&1 | tee log.txt