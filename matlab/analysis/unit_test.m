
result = get_training_curve('C:\Users\v-xiangz\Desktop', {'20160116-131519.12592'}, 256);
[~, h1] = plot_curve(result, 'r');

result = get_training_curve('\\msravcg11\debug_caffepro\rfcnv2_coco_p1n3_lp20_avg_224x_withreg\log', ...
    {'20160116-023444.28404', '20160118-112007.33616'}, 96);
[~, h2] = plot_curve(result, 'b');

result = get_training_curve('\\msravcg03\d$\v-xiangz\debug_caffepro\rfcnv2_coco_p1n3_lp20_avg_224x\log', ...
    {'20160114-175020.34960'}, 96);
[~, h3] = plot_curve(result, 'g');

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\rcnn_coco_p1n3_pad0.2_avg_img224\log', ...
    {'20160111-212733.45196'}, 256);
[~, h4] = plot_curve(result, 'k');

% result = get_training_curve('\\msravcg11\d$\v-xiangz\debug_caffepro\rfcnv2_coco_p1n3_pad0.2_avg\log', ...
%     {'20160111-212733.45196'}, 256);
% plot_curve(result, 'g')

result = get_training_curve('\\msravcg05\d$\v-xiangz\debug_caffepro\rcnn_coco_p1n3_pad0.2_avg_img224_set2\log', ...
    {'20160118-214822.38136'}, 256);
[~, h5] = plot_curve(result, 'm');

result = get_training_curve('\\MSRAVCG10\debug_caffepro\rfcnv2_coco_p1n3_lp16_sp16_avg_224x\log', ...
    {'20160114-213641.55144', '20160116-140118.61540', '20160118-111533.65980'}, 96);
[~, h6] = plot_curve(result, 'c');
legend([h1, h2, h3, h4], 'RFCN with reg', 'RCNN with reg', 'RFCN no reg', 'RCNN no reg')