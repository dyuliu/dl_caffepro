
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20\log', ...
    {'20160308-003317.17640'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain20 (baseline)';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar-plain20-softorth-1e-2-withwc\log', ...
    {'20160419-170051.85056'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'orth-0.1';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar-plain20-softorth-1-withwc\log', ...
    {'20160419-170531.16684'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'orth-1';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar-plain20-softorth-10-withwc\log', ...
    {'20160419-171031.21752'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'orth-10';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-softorth-set2-1-withwc\log', ...
    {'20160419-230410.5756'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'orth-1-set2';

legend(handle, txt)