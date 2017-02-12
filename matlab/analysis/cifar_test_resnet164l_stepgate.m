
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-stepgate-v2\log', ...
    {'20160331-235424.12552'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'stepgate-v2';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-fixedgate0.1\log', ...
    {'20160329-064428.13496'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'fixedgate-0.1';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-householder\log', ...
    {'20160331-235623.8704'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = '3x3-householder';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-stepgate-full\log', ...
    {'20160329-212208.5836'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'stepgate-full';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-stepgate-asynm\log', ...
    {'20160331-085222.10332'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'stepgate-asym';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-shareproj\log', ...
    {'20160405-065645.2416'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'shareproj';

legend(handle, txt)