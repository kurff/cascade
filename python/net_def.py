import sys
sys.path.append("./caffe-cascade/distribute/python");
import caffe 
from caffe import layers as L
from caffe import params as P
from caffe import model_libs
net = caffe.NetSpec();
net.data, net.label = L.TextData(data_param=dict(source="lmdb",backend=P.Data.LMDB,batch_size=4), ntop=2)

net.proposal = L.TextProposal(net.data, text_proposal_param=dict(num_proposals=200, min_size=20, proposal_method=["Canny","MSER"]));
net.data_process = L.TransformData(net.data, transform_param=dict(mean_value=[104,117,123]));



model_libs.VGGNetBodyHalf(net, from_layer='data_process',need_fc=False, fully_conv=True, reduced=True, dilated=True,dropout=False);
#model_libs.ResNet101Body(net, from_layer='data')

net.feat1 = L.ROIPooling(net.pool1,net.proposal, roi_pooling_param=dict(pooled_h=7, pooled_w=7, spatial_scale=0.5));
net.feat2 = L.ROIPooling(net.pool2,net.proposal, roi_pooling_param=dict(pooled_h=7, pooled_w=7, spatial_scale=0.25));
net.feat3 = L.ROIPooling(net.pool3,net.proposal, roi_pooling_param=dict(pooled_h=7, pooled_w=7, spatial_scale=0.125));
net.feat4 = L.ROIPooling(net.pool4,net.proposal, roi_pooling_param=dict(pooled_h=7, pooled_w=7, spatial_scale=0.0625));
net.feat5=  L.ROIPooling(net.relu5_3,net.proposal, roi_pooling_param=dict(pooled_h=7, pooled_w=7, spatial_scale=0.03125));

net.feat = L.Concat(net.feat1, net.feat2, net.feat3, net.feat4);

net.fc = L.InnerProduct(net.feat,num_output=128);
net.drop6 = L.Dropout(net.fc, dropout_ratio=0.5, in_place=True)
net.logit = L.InnerProduct(net.fc,num_output=2);
net_param = net.to_proto()

net.loss = L.SoftmaxWithLoss(net.logit, net.label);


with open("net.prototxt", 'w') as f:
  f.write('{}'.format(net.to_proto()))



