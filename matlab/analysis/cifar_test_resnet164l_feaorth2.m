
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\bottleneck164-feaorth-scale0.1\log', ...
    {'20160505-002356.90304'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-scale0.1';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-fixedgate0.1\log', ...
    {'20160329-064428.13496'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'fixedgate-0.1';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\bottleneck164-feaorth-set2-scale0.1\log', ...
    {'20160505-002617.102104'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-scale0.1-set2';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\bottleneck164-feaorth-set3-scale0.1\log', ...
    {'20160505-002821.91516'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-scale0.1-set3';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\bottleneck164-async-scale0.1\log', ...
    {'20160506-005107.102904'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-scale0.1-async';
legend(handle, txt)