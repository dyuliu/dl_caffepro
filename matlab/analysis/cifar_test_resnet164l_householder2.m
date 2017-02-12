
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3-order16-multi2\log', ...
    {'20160408-083508.9168'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-order16-multi2';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3-order8-multi4\log', ...
    {'20160409-020448.8292'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-order8-multi4';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3res-order16\log', ...
    {'20160409-024742.3268'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'orth3x3res-order16';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3-order16\log', ...
    {'20160406-224927.15116'}, 1);
[~, h] = plot_curve(result, [0.5, 0.9, 0.2]);
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-order16';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3add-order16-4\log', ...
    {'20160411-040235.21256'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-add-order16-4';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3add-order8-4\log', ...
    {'20160411-035929.9456'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-add-order8-4';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3add-order16-2\log', ...
    {'20160411-041441.2592'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-add-order16-2';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3add-order8-2\log', ...
    {'20160411-035649.14244'}, 1);
[~, h] = plot_curve(result, [1, 0.3, 0.7]);
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-add-order8-2';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3add-order4-8\log', ...
    {'20160411-042038.21668'}, 1);
[~, h] = plot_curve(result, [0.2, 0.8, 0.7]);
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-add-order4-8';

legend(handle, txt)