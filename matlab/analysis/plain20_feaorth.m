
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20\log', ...
    {'20160308-003317.17640'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain20 (baseline)';

result = get_training_curve('\\msravcg07\d$\v-xiangz\cifar-plain20-feaorth\log', ...
    {'20160427-205635.12424'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-1';

result = get_training_curve('\\msravcg07\d$\v-xiangz\cifar-plain20-feaorth-coeff0.1\log', ...
    {'20160427-211404.9000'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-0.1';

result = get_training_curve('\\msravcg07\d$\v-xiangz\cifar-plain20-feaorth-coeff0.01\log', ...
    {'20160427-211847.6408'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-0.01';

result = get_training_curve('\\msravcg07\d$\v-xiangz\cifar-plain20-feaorth-coeff0.001\log', ...
    {'20160427-212500.19076'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-0.001';

result = get_training_curve('\\msravcg12\d$\v-xiangz\debug_caffepro\cifar-plain20-feaorth-set3\log', ...
    {'20160428-001813.17376'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'feaorth-set3';

legend(handle, txt)