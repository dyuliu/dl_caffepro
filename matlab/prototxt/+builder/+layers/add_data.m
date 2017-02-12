function result = add_data(name, top, source, mean, batch_size, color_kl, template, gpus)
    scale_jitter_type = [];
    crop_ratio_upperbound = [];
    crop_ratio_lowerbound = [];
    cache_data = true;

    if strcmp(template, 'cifar-train')
        batch_img_size = 32;
        crop_type = 1;
        random_shuffle = true;
        crop_ratio = 0.8;
    elseif strcmp(template, 'cifar-test')
        batch_img_size = 32;
        crop_type = 3;
        random_shuffle = false;
        crop_ratio = 0.8;
    elseif strcmp(template, 'imagenet-train-inceptionv4')
        batch_img_size = 299;
        crop_type = 1;
        random_shuffle = true;
        crop_ratio = 0.875;
        cache_data = false;
        scale_jitter_type = struct('value', 'UniAreaV2');
        crop_ratio_upperbound = 1.0;
        crop_ratio_lowerbound = 0.08;
    elseif strcmp(template, 'imagenet-train')
        batch_img_size = 224;
        crop_type = 1;
        random_shuffle = true;
        crop_ratio = 0.875;
        cache_data = false;
        scale_jitter_type = struct('value', 'UniRatio');
        crop_ratio_upperbound = 0.875;
        crop_ratio_lowerbound = 0.467;
    elseif strcmp(template, 'imagenet-test')
        batch_img_size = 224;
        crop_type = 3;
        random_shuffle = false;
        crop_ratio = 0.875;
    elseif strcmp(template, 'imagenet-test-inceptionv4')
        batch_img_size = 299;
        crop_type = 3;
        random_shuffle = false;
        crop_ratio = 0.875;
    end
    
    gpu_split = [];
    if ~isempty(gpus)
        assert(size(gpus, 2) == 2);
        gpu_split = cell(1, 4 * size(gpus, 1));
        
        for i = 1 : size(gpus, 1)
            gpu_split{i * 2 - 1} = 'split_minibatch';
            gpu_split{i * 2} = gpus(i, 1);
        end
        
        for i = 1 : size(gpus, 1)
            gpu_split{i * 2 - 1 + size(gpus, 1) * 2} = 'split_gpu_id';
            gpu_split{i * 2 + size(gpus, 1) * 2} = gpus(i, 2);
        end
    end
    
    result = builder.layers.add_layer(name, 'data_bigfile', {}, top, ...
        {'data_bigfile_param', { ...
            'source', source, ...
            'mean_file', mean, ...
            'channel_num', 3, ...
            'batch_size', batch_size, ...
            'batch_img_size', batch_img_size, ...
            'crop_type', crop_type, ...
            'crop_ratio', crop_ratio, ...
            'cache_data', iif(cache_data, 'true', 'false'), ...
            'random_shuffle', iif(random_shuffle, 'true', 'false'), ...
            'scale_jitter_type', scale_jitter_type, ...
            'crop_ratio_upperbound', crop_ratio_upperbound, ...
            'crop_ratio_lowerbound', crop_ratio_lowerbound, ...
            'color_kl_dir', color_kl
            } ...
        }, ...
        'gpu_split', gpu_split ...
        );
end

