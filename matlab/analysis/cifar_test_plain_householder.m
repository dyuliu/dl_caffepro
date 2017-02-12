
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20\log', ...
    {'20160308-003317.17640'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain20 (baseline)';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder\log', ...
    {'20160330-060400.19824'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain40-householder\log', ...
    {'20160330-063015.6892'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'plain40-householder';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-order2\log', ...
    {'20160330-070246.4576'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order2';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-nowc\log', ...
    {'20160331-004133.14792'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-nowc';

legend(handle, txt)