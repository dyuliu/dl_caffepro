
handle = [];
txt = {};

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain54\log', ...
    {'20160307-063444.7288'}, 1);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'plain54 (baseline)';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\cifar-plain54_resweight\log', ...
    {'20160307-064147.15888'}, 1);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'plain54-resweight';


legend(handle, txt)