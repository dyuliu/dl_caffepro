function result = get_submission_format(data, d_ncls, d_thres)
%GET_SUBMISSION_FORMAT Convert box data to the format for evaluation
%   data: an array of structures contains box sizes, labels and confs
%   d_ncls: number of classes
%   d_thres: optional. only boxes of conf > thres will be processed
    if nargin < 3, d_thres = 0; end;
    
    n_pics = length(data);
    
    result = cell(d_ncls, 1);
    for i = 1 : d_ncls
        result{i} = cell(n_pics, 1);
    end
    
    for i = 1 : length(data) 
        boxes = data(i).sizes + 1;
        labels = data(i).labels + 1;
        confs = data(i).confs;
        
        I = confs > d_thres;
        boxes = boxes(I, :);
        labels = labels(I);
        confs = confs(I);
        
        for j = 1 : length(confs)
            if confs(j) > d_thres
                lb = labels(j);
                if lb <= d_ncls
                    result{lb}{i} = [result{lb}{i}; double(boxes(j, :)), confs(j)];
                end
            end
        end
    end
end

