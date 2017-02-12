
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

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder6_normwc\log', ...
    {'20160405-084144.6776'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order6';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder12_normwc\log', ...
    {'20160405-083538.20580'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order12';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder24_normwc\log', ...
    {'20160405-083311.2588'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order24';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder36_normwc\log', ...
    {'20160405-083844.10504'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order36';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder48_normwc\log', ...
    {'20160405-084453.13212'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order48';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder36_normwc_seq\log', ...
    {'20160406-084202.11248'}, 1);
[~, h] = plot_curve(result, [0.8, 0.8, 0.3]);
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order36-seq';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder24-48-96_normwc\log', ...
    {'20160405-205003.9900'}, 1);
[~, h] = plot_curve(result, [0.3, 0.8, 0.8]);
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order24-48-96';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-householder-higherorder32-64-128_normwc\log', ...
    {'20160405-205339.19360'}, 1);
[~, h] = plot_curve(result, [0.3, 0.8, 0.3]);
handle(end + 1) = h;
txt{end + 1} = 'plain20-householder-order32-64-128';

legend(handle, txt)