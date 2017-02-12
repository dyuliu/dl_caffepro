
handle = [];
txt = {};

result = get_training_curve('D:\cifar_experiments\_EXTEND_SET2_bn_cifar_basic_110_2gpu_4stg_optionA_wc\log', ...
    {'20160218-154938.2152'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\resnet101-feaorth\log', ...
    {'20160428-002723.17016'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'resnet101-feaorth';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\resnet101-feaorth-set2\log', ...
    {'20160428-003240.932'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'resnet101-feaorth-set2';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\resnet101-feaorth-set3\log', ...
    {'20160428-165220.14312'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'resnet101-feaorth-set3';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\resnet101-feaorth-set5\log', ...
    {'20160428-172057.12684'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'resnet101-feaorth-set5';

legend(handle, txt)