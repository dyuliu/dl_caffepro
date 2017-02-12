
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-longtrain-narrow\log', ...
    {'20160421-022023.8360'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain20 narrow (baseline)';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain60-longtrain-narrow\log', ...
    {'20160421-021657.7340'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain60 narrow (baseline)';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain60-softorth-longtrain-narrow\log', ...
    {'20160421-023835.8420'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'plain60 narrow softorth';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-softorth-longtrain-narrow\log', ...
    {'20160421-023252.8592'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'plain20 narrow softorth';

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\cifar-plain60-addw-longtrain-narrow\log', ...
    {'20160422-001946.41840'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'plain60 narrow addw';

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\cifar-plain60-addi-longtrain-narrow\log', ...
    {'20160422-150704.6352'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'plain60 narrow addi';

result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\cifar-plain60-softorth-prelu-narrow\log', ...
    {'20160423-005252.25676'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'plain60 narrow addw-softorth';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain60-adddiag-longtrain-narrow\log', ...
    {'20160425-024455.15164'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'plain60 narrow adddiag';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain60-adddiag-softorth-longtrain-narrow\log', ...
    {'20160425-031316.17676'}, 1);
[~, h] = plot_curve(result, [0.8, 0.6, 0.2]);
handle(end + 1) = h;
txt{end + 1} = 'plain60 narrow softorth-adddiag';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain60-adddiag-softorth-varbn-longtrain-narrow\log', ...
    {'20160425-044614.16436'}, 1);
[~, h] = plot_curve(result, [0.2, 0.6, 0.9]);
handle(end + 1) = h;
txt{end + 1} = 'plain60 narrow softorth-varbn';
legend(handle, txt)