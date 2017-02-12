
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20\log', ...
    {'20160308-003317.17640'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain20 (baseline)';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\cifar-plain20-nest\log', ...
    {'20160413-174856.13004'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain20-nesterov';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout-linear0.5-longtrain\log', ...
    {'20160411-231726.15336'}, 1);
[~, h] = plot_curve(result, [1, 0.8, 0.2]);
handle(end + 1) = h;
txt{end + 1} = 'linear0.5-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-baseline-longtrain\log', ...
    {'20160411-002825.21176'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'baseline-longtrain';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-baseline-longtrain-nesterov\log', ...
    {'20160413-033421.26320'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'baseline-longtrain-nesterov';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-resnet164-learnable-dropout-linear0.5-longtrain-nesterov\log', ...
    {'20160413-033656.8784'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'linear0.5-longtrain-nesterov';

legend(handle, txt)