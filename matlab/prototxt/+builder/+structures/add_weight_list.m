function [result, output_weight_names] = add_weight_list(prefix, bottom, num, dims, opt)
    assert(nargin >= 4);
    if nargin <= 4, opt = []; end;
    
    result = '';
    output_weight_names = cell(1, num);
    for i = 1 : num
        output_weight_names{i} = [prefix, '_weightsrc_', num2str(i)];
        result = cat(2, result, builder.layers.add_weight(output_weight_names{i}, bottom, output_weight_names{i}, dims, opt));
    end
end

