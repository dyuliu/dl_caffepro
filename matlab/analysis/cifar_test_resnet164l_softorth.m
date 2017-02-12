
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\cifar-resnet164-softorth-1-withwc\log', ...
    {'20160419-174417.42196'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'orth-1';

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\cifar-resnet164-softorth-1e-1-withwc\log', ...
    {'20160419-182328.41368'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'orth-0.1';

legend(handle, txt)