
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

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder2\log', ...
    {'20160405-032925.11516'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order2';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder6\log', ...
    {'20160405-033344.6240'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order6';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder12\log', ...
    {'20160405-033900.15584'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order12';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder24\log', ...
    {'20160405-034309.15764'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order24';

legend(handle, txt)