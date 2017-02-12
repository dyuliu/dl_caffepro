function imdb = imdb_from_coco(root_dir, image_set, flip)
% imdb = imdb_from_voc(root_dir, image_set, year)
%   Builds an image database for the PASCAL VOC devkit located
%   at root_dir using the image_set and year.
%
%   Inspired by Andrea Vedaldi's MKL imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

%imdb.name = 'coco_train_2014'
%imdb.image_dir = '/work4/rbg/VOC2007/VOCdevkit/VOC2007/JPEGImages/'
%imdb.extension = '.jpg'
%imdb.image_ids = {'000001', ... }
%imdb.sizes = [numimages x 2]
%imdb.classes = {'aeroplane', ... }
%imdb.num_classes
%imdb.class_to_id
%imdb.class_ids
%imdb.eval_func = pointer to the function that evaluates detections
%imdb.roidb_func = pointer to the function that returns regions of interest

if nargin < 3
    flip = false;
end

% cache_file = ['./imdb/cache/imdb_coco_' image_set];
% if flip
%     cache_file = [cache_file, '_flip'];
% end

addpath(sprintf('%s/MatlabAPI/', root_dir));


% try
%   load(cache_file);
% catch
  annFile=sprintf('%s/annotations/instances_%s.json',root_dir,image_set);
  cocoAPI=CocoApi(annFile);

  imdb.name = ['coco_' image_set];
  
  switch image_set    
      case {'test-dev2015'}
          image_folder = 'test2015';
      otherwise
          image_folder = image_set;
  end
  
  imdb.image_dir = sprintf('%s/images/%s/', root_dir, image_folder);
  imdb.image_ids_coco = cocoAPI.getImgIds();
  img_info = cocoAPI.loadImgs(cocoAPI.getImgIds());
  [~, imdb.image_ids, ~] = cellfun(@(x)  fileparts(x), extractfield(img_info, 'file_name'), 'UniformOutput', false);
  
  switch image_set    
      case {'test-dev2015'}
          imdb.image_ids = cellfun(@(x) regexprep(x, 'test-dev2015', 'test2015'), imdb.image_ids, 'UniformOutput', false);
      otherwise
  end
  
  imdb.sizes(:, 1) = extractfield(img_info, 'height');
  imdb.sizes(:, 2) = extractfield(img_info, 'width');
  imdb.extension = 'jpg';
  imdb.flip = flip;
  if flip
      parfor i = 1:length(imdb.image_ids)
          cur_image =sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
          cur_flip_image = sprintf('%s/%s_flip.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);
          if ~exist(cur_flip_image, 'file')
             im = imread(cur_image);
             imwrite(fliplr(im), cur_flip_image);
          end
      end
      img_num = length(imdb.image_ids)*2;
      image_ids_coco = imdb.image_ids_coco;
      imdb.image_ids_coco(1:2:img_num) = image_ids_coco;
      imdb.image_ids_coco(2:2:img_num) = image_ids_coco;
      image_ids = imdb.image_ids;
      imdb.image_ids(1:2:img_num) = image_ids;
      imdb.image_ids(2:2:img_num) = cellfun(@(x) [x, '_flip'], image_ids, 'UniformOutput', false);
      imdb.flip_from = zeros(img_num, 1);
      imdb.flip_from(2:2:img_num) = 1:2:img_num;
      
      sizes = imdb.sizes;
      imdb.sizes(1:2:img_num, :) = sizes;
      imdb.sizes(2:2:img_num, :) = sizes;
  end
  
  imdb.coco_class_ids = cocoAPI.getCatIds();
  imdb.classes = extractfield(cocoAPI.loadCats(imdb.coco_class_ids), 'name');
  imdb.num_classes = length(imdb.classes);
  imdb.class_ids = 1:imdb.num_classes;
  imdb.coco_id_to_id = containers.Map(imdb.coco_class_ids, 1:imdb.num_classes);
  imdb.id_to_coco_id = containers.Map(1:imdb.num_classes, imdb.coco_class_ids);

  % private COCO details
  imdb.details.cocoAPI = cocoAPI;
  imdb.details.coco_opts.resdir = [root_dir '/results/'];
  imdb.details.coco_opts.max_per_image = 100;
  imdb.details.coco_opts.test_set = image_set;

  % VOC specific functions for evaluation and region of interest DB
  imdb.eval_func = @imdb_eval_coco;
  imdb.roidb_func = @roidb_from_coco;
  imdb.image_at = @(i) sprintf('%s/%s.%s', imdb.image_dir, imdb.image_ids{i}, imdb.extension);

%   fprintf('Saving imdb to cache...');
%   save(cache_file, 'imdb', '-v7.3');
%   fprintf('done\n');
% end
