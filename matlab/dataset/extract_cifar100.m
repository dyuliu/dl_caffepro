function extract_cifar100(s_filename, s_dest_folder, s_fmt, i_pad)
%EXTRACT_CIFAR100 extract cifar-100 dataset into separated image files
%   s_filename: cifar-100 binary dataset file
%   s_dest_folder: dest_folder to save
%   s_fmt: save image format
%   i_pad: padding width

    [labels, imgs] = load_cifar100_bin(s_filename);
    pad_img = uint8(zeros(32 + i_pad * 2, 32 + i_pad * 2, 3));
    pad_img(:, :, 1) = 125;
    pad_img(:, :, 2) = 123;
    pad_img(:, :, 3) = 114;
    
    num_images = zeros(1, 100);
    
    if ~isempty(s_fmt) && s_fmt(1) ~= '.', s_fmt = strcat('.', s_fmt); end;
    
    for i = 1 : length(labels)
        label = labels(i);
        
        img = imgs{i};
        save_img = pad_img;
        save_img(i_pad + 1 : i_pad + 32, i_pad + 1 : i_pad + 32, :) = img;
        
        num_images(label + 1) = num_images(label + 1) + 1;
        
        
        image_name = strcat(num2str(num_images(label + 1)), s_fmt);
        dir_name = strcat(s_dest_folder, '\', num2str(label));
        
        if ~exist(dir_name, 'dir'), system(['mkdir ', dir_name]); end;
        
        imwrite(save_img, fullfile(dir_name, image_name));   
    end
end

