
handle = [];
txt = {};

result = get_training_curve('\\msravcg05\d$\kahe\pro_SET2_bn_bottleneck_IncResNet101_v1_8gpu_4stg\log', ...
    {'20160220-230707.13260'}, 256);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\pro_SET2_bn_bottleneck_IncResNet101_v1_8gpu_4stg_debug\log', ...
    {'20160311-124330.9348'}, 256);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'baseline-fastupdate';
