
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20\log', ...
    {'20160308-003317.17640'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain20 (baseline)';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20_resweight\log', ...
    {'20160308-003630.12396'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain20-resweight';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20_resweight_coeff55\log', ...
    {'20160308-052754.19528'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'plain20-resweight-coeff55';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20_transweight\log', ...
    {'20160309-011352.9608'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'plain20-transweight';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20_transweight_coeffsqrt\log', ...
    {'20160309-015227.10964'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'plain20-transweight-coeffsqrt';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20_transweight_largelr\log', ...
    {'20160309-020923.14300'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'plain20-transweight-lr0.5';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-largelr\log', ...
    {'20160309-021657.11224'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'plain20-lr0.5 (baseline)';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain20_transweight_largelr1\log', ...
    {'20160309-182825.74520'}, 1);
[~, h] = plot_curve(result, [0.8, 0.8, 1]);
handle(end + 1) = h;
txt{end + 1} = 'plain20-transweight-lr1';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain20-largelr1\log', ...
    {'20160309-185200.67144'}, 1);
[~, h] = plot_curve(result, [1, 0.8, 0.2]);
handle(end + 1) = h;
txt{end + 1} = 'plain20-lr1 (baseline)';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain20_transweight_coeffsqrt_largelr1\log', ...
    {'20160309-192006.77888'}, 1);
[~, h] = plot_curve(result, [0.8, 0.1, 0.8]);
handle(end + 1) = h;
txt{end + 1} = 'plain20-transweight-coeffsqrt-lr1';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar\cifar-plain20_transweight_coeffsqrt_largelr0.5\log', ...
    {'20160309-205350.77204'}, 1);
[~, h] = plot_curve(result, [0.1, 0.8, 0.8]);
handle(end + 1) = h;
txt{end + 1} = 'plain20-transweight-coeffsqrt-lr0.5';
legend(handle, txt)