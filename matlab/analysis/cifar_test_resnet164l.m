
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-resweight\log', ...
    {'20160308-044233.19908'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'resweight';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-resweight-coeff55\log', ...
    {'20160308-050302.17900'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'resweight-coeff55';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-resweight-coeff55-fastupdate\log', ...
    {'20160310-203914.15364'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'resweight-coeff55-fastupdate';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-resweight-coeff55-sliding-val\log', ...
    {'20160311-040327.13340'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'resweight-coeff55-fastupdate-sliding';

legend(handle, txt)