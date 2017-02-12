
handle = [];
txt = {};

result = get_training_curve('\\msravcg01\d$\kahe\pro_SET2_bn_bottleneck_IncResNet101_v3apool1_8gpu_4stg_uniareav2_slidehack\log', ...
    {'20160410-082820.15008'}, 256);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\pro_SET2_bn_bottleneck_IncResNet101_v3apool1_8gpu_4stg_uniareav2_slidehack_ssgd_m4\log', ...
    {'20160517-214834.5928'}, 256 * 4);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'ssgd';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\pro_SET2_bn_bottleneck_IncResNet101_v3apool1_8gpu_4stg_uniareav2_slidehack_lrx4\log', ...
    {'20160415-010348.3736', '20160419-231127.5916'}, 256 * 4);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'asgd-lrx4-longtrain';

legend(handle, txt)