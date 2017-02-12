
handle = [];
txt = {};

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain27\log', ...
    {'20160307-141949.65012'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain27 (baseline)';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain27_resweight\log', ...
    {'20160307-170100.73072'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain27-resweight';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain27_corrimpl\log', ...
    {'20160307-155414.68464'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'plain27 (baseline, impl. 2)';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain27_resweight_wnorm\log', ...
    {'20160307-175904.65628'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'plain27-resweight-outnorm';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain27_resweight_swnorm\log', ...
    {'20160307-184121.60252'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'plain27-resweight-innorm';

legend(handle, txt)