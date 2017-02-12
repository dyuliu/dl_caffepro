
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc_cluster4\log', ...
    {'20160131-185359.2580'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'end2end-co0.1-center20-bn';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc_cluster4_norm_online\log', ...
    {'20160202-185548.29592'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'periodic(100,200)-co0.1-center20-l2norm';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc_cluster4_norm_online_set2\log', ...
    {'20160202-233609.48060'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'periodic(400,1000)-co0.1-center20-l2norm';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc_cluster_norm_online\log', ...
    {'20160202-171854.50248'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'periodic(100,200)-co1-center20-l2norm';

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc_cluster4_online\log', ...
    {'20160202-170031.9916'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'periodic(100,200)-co0.1-center20-bn';


legend(handle, txt)