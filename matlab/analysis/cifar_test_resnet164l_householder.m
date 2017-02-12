
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-householder\log', ...
    {'20160331-235623.8704'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'orth3x3';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orthproj\log', ...
    {'20160405-224857.15660'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'orthproj';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orthprojall\log', ...
    {'20160405-233339.8020'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'orthall';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orthprojall-order6\log', ...
    {'20160405-234319.3952'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'orthall-order6';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orthprojall-order16\log', ...
    {'20160406-002811.11600'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'orthall-order16';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orthproj-order6\log', ...
    {'20160405-230818.7132'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'orthproj-order6';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orthproj-order16-32-64\log', ...
    {'20160406-003616.1320'}, 1);
[~, h] = plot_curve(result, [0.8, 0.8, 0.2]);
handle(end + 1) = h;
txt{end + 1} = 'orthproj-order16-32-64';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orthprojall-order16-32-64\log', ...
    {'20160406-003045.8480'}, 1);
[~, h] = plot_curve(result, [0.2, 0.8, 0.8]);
handle(end + 1) = h;
txt{end + 1} = 'orthall-order16-32-64';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-shareproj\log', ...
    {'20160405-065645.2416'}, 1);
[~, h] = plot_curve(result, [0.2, 0.8, 0.3]);
handle(end + 1) = h;
txt{end + 1} = 'share-proj';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orthproj-order6_seq\log', ...
    {'20160406-084700.16336'}, 1);
[~, h] = plot_curve(result, [0.5, 0.5, 0.5]);
handle(end + 1) = h;
txt{end + 1} = 'orthproj-order6-seq';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-orth3x3-order16\log', ...
    {'20160406-224927.15116'}, 1);
[~, h] = plot_curve(result, [0.5, 0.9, 0.2]);
handle(end + 1) = h;
txt{end + 1} = 'orth3x3-order16';

legend(handle, txt)