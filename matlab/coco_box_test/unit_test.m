
initenv;

% imdb = imdb_from_coco('D:\Data\coco', 'val2014');
% reg_boxes = load_locfea('D:\rfcn_experiments\models\rcnn_coco_p1n3_pad0.2_avg_img224_withreg\save_val.full.locfea', imdb.image_ids, '.jpg');
reg_boxes = load_locclsfea('D:\rfcn_experiments\models\rcnn_coco_p1n3_pad0.2_avg_img224_withreg_cluster\save_val.locclsfea', imdb.image_ids, '.jpg');

% seed_rand(6);
% plot_id = randperm(length(reg_boxes), 1000);
% 
% threshold = 0.7;
% for j = 1 : length(plot_id) 
%     i = plot_id(j);
%     
%     pic_file = fullfile('D:\Data\coco\images\test-dev2015', reg_boxes(i).picname);
%     im = imread(pic_file);
%     
%     
%     boxes = reg_boxes(i).confs > threshold;
%     confs = reg_boxes(i).confs(boxes);
%     labels = reg_boxes(i).labels(boxes) + 1;
%     boxes = reg_boxes(i).sizes(boxes, :) + 1;
%     
%     box_conf = [double(boxes), confs];
%     view = cell(80, 1);
%     for lb = 1 : 80
%         index = (labels == lb);
%         view{lb} = box_conf(index, :);
%         view{lb} = view{lb}(nms(view{lb}, 0.3), :);
%     end
%     
%     im = showboxes(im, view, imdb.classes, 'default', true);
%     imwrite(im, ['images\', reg_boxes(i).picname]);
% %     
% %     pause
% end

submit_result = get_submission_format(reg_boxes, 80, 0.1);
res = imdb_eval_coco(submit_result, imdb, 'cache', 'test', true);