
handle = [];
txt = {};

result = get_training_curve('\\msravcg10\d$\v-xiangz\debug_caffepro\SET2_bn_bottleneck_v1_dp5_8gpu_4stg\log', ...
    {'20151203-190444.6320'}, 256);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_sgdfaster_debug\log', ...
    {'20160310-211114.63188'}, 256);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'baseline-fastupdate';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_sgdfaster_cudnn4_debug\log', ...
    {'20160323-184742.73456'}, 256);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'baseline-cudnn4';