
r1 = builder.layers.add_conv('conv1', 'data', 'conv1_t', 3, 16, 2, 1, false, struct('weight_filler', {{'type', 'xiangyu'}}));
r2 = builder.layers.add_bn('bn_conv1', 'conv1_t', 'conv1', 'NOT_CALC');
builder.layers.add_layer('conv1_relu', 'relu', {'conv1', 'conv2'}, 'conv1');
builder.layers.add_fc('fc10', 'gpu_concated_pool5', 'fc10', 10, false, struct('bias_filler', {{'type', 'gaussion', 'std', 0.02}}));
builder.layers.add_pool('pool5', 'ins4b17', 'pool5', 'AVE', 8, 1, 1);
builder.layers.add_gpuconcat('GPUConcat', 'pool5', 'gpu_concated_pool5', 6);
builder.layers.add_eltsum('sum', {'input1', 'inputs2'}, 'output', [2, 3.]);
builder.layers.add_softmaxloss('loss', ['fc10', 'label'], 'loss', 0);
builder.layers.add_concat('inp3c', {'input1', 'input2', 'input3'}, {'output1'});
builder.layers.add_weight('w1', 'data', 'w1', [3, 3, 3, 16], struct('weight_filler', {{'type', 'xiangyu'}}));
builder.layers.add_correlation('co1', {'data', 'w1'}, 'cot', 1, 2, false);
builder.layers.add_data('ImageNet', {'data', 'label'}, 'F:\\Data\\ILSVRC2012_img_val_raw_big', 'ImageNet_1000_scale224_mean.xml', 128, [], 'imagenet-test', [64, 0; 64, 1]);
[r1 r2]