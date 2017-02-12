
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\bottleneck164-feaorth\log', ...
    {'20160502-230744.19184'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'feaorth';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\bottleneck164-feaorth-set2\log', ...
    {'20160502-231448.20400'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-set2';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\bottleneck164-feaorth-set3\log', ...
    {'20160502-231855.24692'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-set3';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\bottleneck164-feaorth-set3-v2\log', ...
    {'20160502-232443.23032'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-set3v2';

legend(handle, txt)