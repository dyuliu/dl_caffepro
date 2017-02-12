
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-baseline-longtrain\log', ...
    {'20160411-002825.21176'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'baseline-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout-linear0.5-longtrain\log', ...
    {'20160411-231726.15336'}, 1);
[~, h] = plot_curve(result, [1, 0.8, 0.2]);
handle(end + 1) = h;
txt{end + 1} = 'randdepth-linear0.5-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-droppath\log', ...
    {'20160415-015752.21192'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'droppath-0.5';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-droppath-linear0.5\log', ...
    {'20160415-054556.13128'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'droppath-linear-0.5';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-droppath-longtrain-linear0.5\log', ...
    {'20160415-054837.18344'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'droppath-linear-0.5-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-droppath-longtrain\log', ...
    {'20160415-053352.15356'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'droppath-0.5-longtrain';


legend(handle, txt)