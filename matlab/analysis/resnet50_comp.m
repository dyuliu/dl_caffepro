
handle = [];
txt = {};

result = get_training_curve('\\msravcg10\d$\v-xiangz\debug_caffepro\SET2_bn_bottleneck_v1_dp5_8gpu_4stg\log', ...
    {'20151203-190444.6320'}, 256);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_rank-ohem\log', ...
    {'20160520-132000.1924'}, 256);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'rank-ohem';

result = get_training_curve('\\msr-hdp\Scratch\GW0013\kahe\experiments\pro_SET2_bn_bottleneck_ResNet50_8gpu_4stg_uniareav2_slidehack\log', ...
    {'20160505-201458.13016'}, 256);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'uniarea2-baseline';

result = get_training_curve('\\msravcg01\d$\kahe\pro_SET2_bn_bottleneck_ResNet50_8gpu_4stg_uniareav2_slidehack_rankohem256\log', ...
    {'20160524-143655.6244'}, 256);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'uniarea2-rankohem';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\bottleneck50l_runavg\log', ...
    {'20160524-193132.18776'}, 256);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'runavgbn';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\bottleneck50l_runavg_notcalc\log', ...
    {'20160524-092706.16244'}, 256);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'runavgbn-notcalc';

legend(handle, txt)