
handle = [];
txt = {};

result = get_training_curve('\\MSRAVCG12\debug_caffepro\cifar\_EXTEND_SET2_bn_cifar_bottleneck_164_2gpu_4stg_wc\log', ...
    {'20160128-160719.9704'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout\log', ...
    {'20160407-044835.5296'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'learnable';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout-fixed0.5\log', ...
    {'20160407-073045.20208'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'fixed-0.5';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout-fixed0.8\log', ...
    {'20160407-211432.10516'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'fixed-0.8';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-baseline-longtrain\log', ...
    {'20160411-002825.21176'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'baseline-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout-fixed0.5-longtrain\log', ...
    {'20160411-002825.15188'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'fixed-0.5-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout-fixed0.8-longtrain\log', ...
    {'20160411-002826.26128'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'fixed-0.8-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout-linear0.5-longtrain\log', ...
    {'20160411-231726.15336'}, 1);
[~, h] = plot_curve(result, [1, 0.8, 0.2]);
handle(end + 1) = h;
txt{end + 1} = 'linear0.5-longtrain';

legend(handle, txt)