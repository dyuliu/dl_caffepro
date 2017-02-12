function result = get_training_curve(s_log_folder, cs_log_files, n_unit)
%GET_TRAINING_CURVE Get the training and test error from several log
%files
%   s_log_folder: a string specify the log folder
%   cs_log_files: cell of strings, path of log files
%   d_unit: the unit of x-axis

    [x_train, y_train, x_test, y_test] = cellfun(@(x) from_file(fullfile(s_log_folder, x)), cs_log_files, 'UniformOutput', false);
    max_x_train = -inf; max_x_test = -inf;
    
    res_train_x = []; res_train_y = []; res_test_x = []; res_test_y = [];
    for i = 1 : length(x_train)
        index = x_train{i} > max_x_train;
        res_train_x = cat(2, res_train_x, x_train{i}(index));
        res_train_y = cat(2, res_train_y, y_train{i}(index));
        
        index = x_test{i} > max_x_test;
        res_test_x = cat(2, res_test_x, x_test{i}(index));
        res_test_y = cat(2, res_test_y, y_test{i}(index));
        
        if ~isempty(res_train_x), max_x_train = res_train_x(end); end
        if ~isempty(res_test_x), max_x_test = res_test_x(end); end
    end
    
    result.x_train = res_train_x * n_unit; result.y_train = res_train_y;
    result.x_test = res_test_x * n_unit; result.y_test = res_test_y;
end

function [x_train, y_train, x_test, y_test] = from_file(s_log_file)
    fid = fopen(s_log_file, 'rt');
    data = fread(fid, [1, inf], 'uint8=>char');
    fclose(fid);
    
    train = regexp(data, '--TRAIN:\s*iter = (\d+), error = ([\d\.]+)', 'tokens');
    x_train = cellfun(@(x) str2double(x{1}), train);
    y_train = cellfun(@(x) str2double(x{2}), train);
    
    test = regexp(data, '--TEST:\s*iter = (\d+), error = ([\d\.]+)', 'tokens');
    x_test = cellfun(@(x) str2double(x{1}), test);
    y_test = cellfun(@(x) str2double(x{2}), test);
end

