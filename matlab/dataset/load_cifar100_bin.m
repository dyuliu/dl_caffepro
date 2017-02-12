function [img_label, img_data] = load_cifar100_bin(s_filename)
%LOAD_BIGFILE load images from a single cifar100-format binary file
%   s_filename: dataset file name
    
    fid = fopen(s_filename, 'rb');
    mat = fread(fid, inf, 'uint8=>uint8');
    fclose(fid);
    
    mat = reshape(mat, 2 + 3072, []);
    
    n_pics = size(mat, 2);
    img_label = double(mat(2, :));
    img_data = cell(1, n_pics);
    
    for i = 1 : n_pics
        img_data{i} = reshape(mat(3 : end, i), 32, 32, 3);
        img_data{i} = permute(img_data{i}, [2, 1, 3]);
    end
end