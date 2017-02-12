
handle = [];
txt = {};

result = get_training_curve('D:\cifar_experiments\_EXTEND_SET2_bn_cifar_basic_110_2gpu_4stg_optionA_wc\log', ...
    {'20160218-154938.2152'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\cifar-res110-softorth-1e-2\log', ...
    {'20160419-235828.6128'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'softorth-0.01';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\cifar-res110-softorth-1e-1\log', ...
    {'20160420-000622.21528'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'softorth-0.1';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\cifar-res110-softorth-1\log', ...
    {'20160420-001102.98896'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'softorth-1';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\cifar-res110-softorth-10\log', ...
    {'20160420-001435.68096'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'softorth-10';

legend(handle, txt)