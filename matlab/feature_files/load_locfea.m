function result = load_locfea(s_filename, cs_order, suffix)
%LOAD_LOCFEA load box file. Note that box size and class id are both
%zero-based
%   s_filename: feature file name (*.locfea)
%   cs_order: optional. A cell which specified the order of returned result
%   suffix: optional. The suffix for cs_order
    if nargin < 2, cs_order = []; end;
    if nargin < 3, suffix = ''; end;

    features = {};
    fid = fopen(s_filename, 'rb');

    while true
        picname = read_string(fid);
        if feof(fid)
            break;
        end
        nbox = fread(fid, 1, 'int32');
        raw_data = fread(fid, [6, nbox], 'int32=>int32');
        
        raw_data = raw_data';
        sz = raw_data(:, 1 : 4);
        label = raw_data(:, 5);
        conf = typecast(raw_data(:, 6), 'SINGLE');
        
        features{end + 1} = struct('picname', picname, 'sizes', sz, 'labels', label, 'confs', conf);
    end
    
    fclose(fid);
    result = cell2mat(features);
    
    if ~isempty(cs_order)
        pic_name_map = containers.Map('KeyType', 'char', 'ValueType', 'double');
        for i = 1 : length(result)
            pic_name_map(result(i).picname) = i;
        end
        
        % check file names
        for i = 1 : length(cs_order) 
            if ~isKey(pic_name_map, [cs_order{i}, suffix])
                disp(cs_order{i})
                pic_name_map([cs_order{i}, suffix]) = 1; % prevent error
            end
        end
        
        order = arrayfun(@(x) pic_name_map([cs_order{x}, suffix]), 1 : length(cs_order));
        result = result(order);
    end
end

