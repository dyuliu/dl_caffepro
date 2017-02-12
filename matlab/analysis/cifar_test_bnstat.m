
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20\log', ...
    {'20160308-003317.17640'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain-20';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-sliding-val\log', ...
    {'20160310-082354.17088'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain-20-movingavg';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-sliding-val-fastupdate\log', ...
    {'20160310-082607.2008'}, 1);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'plain-20-movingavg-fastupdate';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-fastupdate\log', ...
    {'20160310-202459.19436'}, 1);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'plain-20-fastupdate';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-run2\log', ...
    {'20160310-204521.13096'}, 1);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'plain-20-train2';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-run3\log', ...
    {'20160310-232315.7836'}, 1);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'plain-20-train3';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-sliding-val_run2\log', ...
    {'20160311-012518.5164'}, 1);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'plain-20-movingavg-run2';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain20-sliding-val_0.99\log', ...
    {'20160311-015432.5240'}, 1);
[~, h] = plot_curve(result, [0.8, 0.8, 0.1]);
handle(end + 1) = h;
txt{end + 1} = 'plain-20-movingavg-0.99';

legend(handle, txt)