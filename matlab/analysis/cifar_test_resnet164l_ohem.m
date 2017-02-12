
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\bottleneck164-rank-ohem-128,128\log', ...
    {'20160516-183949.107760'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = '128-128';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\bottleneck164-rank-ohem\log', ...
    {'20160516-183203.21272'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = '128-256';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\bottleneck164-rank-ohem-128,512\log', ...
    {'20160516-183941.60104'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = '128-512';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\bottleneck164-rank-ohem-128,1024\log', ...
    {'20160516-184148.93180'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = '128-1024';

legend(handle, txt)