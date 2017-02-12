function [img_names, img_data] = load_bigfile(s_filename)
%LOAD_BIGFILE load images from a single big file
%   s_filename: bigfile name
    
    fid = fopen(s_filename, 'rb');
    
    img_names = {};
    img_data = {};
    
    while true
        name = read_string(fid);
        if feof(fid), break; end;
        
        len = fread(fid, 1, 'int');
        data = fread(fid, [1, len], 'uint8');
        
        [~, ~, fmt] = fileparts(name);
        im = imdecode(data, fmt);
        
        img_names{end + 1} = name;
        img_data{end + 1} = im;
    end 
    
    fclose(fid);
end

