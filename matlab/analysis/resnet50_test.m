
handle = [];
txt = {};

result = get_training_curve('\\msravcg10\d$\v-xiangz\debug_caffepro\SET2_bn_bottleneck_v1_dp5_8gpu_4stg\log', ...
    {'20151203-190444.6320'}, 256);
[~, h] = plot_curve(result, 'b');
handle(end + 1) = h;
txt{end + 1} = 'baseline';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_cluster4_norm_online_set2\log', ...
    {'20160204-180452.6644', '20160218-193350.6488'}, 128);
[~, h] = plot_curve(result, 'r');
handle(end + 1) = h;
txt{end + 1} = 'periodic(400,1000)-co0.1-center1000-l2norm';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_cluster4_norm_online_set2a\log', ...
    {'20160204-180501.10644', '20160218-193628.21568'}, 128);
[~, h] = plot_curve(result, 'g');
handle(end + 1) = h;
txt{end + 1} = 'periodic(400,1000)-co0.1-center2000-l2norm';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_cluster4_norm_online_set2b\log', ...
    {'20160204-180513.2580', '20160218-193819.15620'}, 128);
[~, h] = plot_curve(result, 'm');
handle(end + 1) = h;
txt{end + 1} = 'periodic(400,1000)-co0.1-center500-l2norm';

result = get_training_curve('\\msravcg11\D$\v-xiangz\debug_caffepro\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_cluster4_norm_online_set2c\log', ...
    {'20160217-231234.21060'}, 256);
[~, h] = plot_curve(result, 'c');
handle(end + 1) = h;
txt{end + 1} = 'periodic(400,1000)-co0.1-center4000-l2norm';

result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_cluster4_norm_online_set2d\log', ...
    {'20160217-235108.6136', '20160223-201155.5448'}, 128);
[~, h] = plot_curve(result, 'k');
handle(end + 1) = h;
txt{end + 1} = 'periodic(800,2000)-co0.1-center2000-l2norm';


result = get_training_curve('\\gcr\Scratch\B99\v-xiangz\SET2_bn_bottleneck_v1_dp5_8gpu_4stg_cluster4_norm_online_set2e\log', ...
    {'20160218-003718.7668', '20160223-201305.1644'}, 128);
[~, h] = plot_curve(result, 'y');
handle(end + 1) = h;
txt{end + 1} = 'periodic(2000,5000)-co0.1-center2000-l2norm';

legend(handle, txt)