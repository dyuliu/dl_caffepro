opt = struct();
opt.relu_rename_fun = @(prefix) ['relu', prefix];
builder.structures.add_conv_st('conv1', 'data', 7, 64, 2, 3, false, true, true, 'NOT_CALC', opt)
builder.structures.add_bottleneck_st('5a', 'ins4f', 512, 2048, 2, true, 'NOT_RECORD');