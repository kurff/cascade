import sys
sys.path.append("./caffe-cascade/distribute/python");
import caffe 
from caffe import layers as L
from caffe import params as P
from caffe import model_libs
net = caffe.NetSpec();
net.data, net.label = L.TextData(data_param=dict(source="lmdb",backend=P.Data.LMDB,batch_size=4), ntop=2)


model_libs.VGGNetBody(net, from_layer='data',fully_conv=True, reduced=True, dilated=True,dropout=False);
#model_libs.ResNet101Body(net, from_layer='data')
net_param = net.to_proto()

with open("net.prototxt", 'w') as f:
  f.write('{}'.format(net.to_proto()))