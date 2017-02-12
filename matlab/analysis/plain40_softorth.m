
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20\log', ...
    {'20160308-003317.17640'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain20 (baseline)';

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\cifar-plain40-softorth-1-withwc-longtrain\log', ...
    {'20160419-203649.37732'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain40-orth-1-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain40-longtrain\log', ...
    {'20160420-010333.4652'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'plain40-longtrain (baseline)';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\cifar-plain40-softorth-1-halfbn\log', ...
    {'20160420-205248.53468'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'plain40-orth-1-longtrain-halfbn';

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\cifar-plain40-softorth-1-withwc-longtrain\log', ...
    {'20160419-203649.37732'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain40-orth-1-longtrain';

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\cifar-plain20-softorth-1-withwc-longtrain\log', ...
    {'20160420-184112.41296'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'plain20-orth-1-longtrain';

legend(handle, txt)