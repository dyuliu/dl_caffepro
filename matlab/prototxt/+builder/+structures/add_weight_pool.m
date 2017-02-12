function [result, output_weight_names] = add_weight_pool(prefix, bottom, num, dims, coeff_matrix, global_coeff, opt)
    assert(nargin >= 4);
    if nargin <= 4, coeff_matrix = []; end;
    if nargin <= 5, global_coeff = 1; end;
    if nargin <= 6, opt = []; end;
    
    if isempty(coeff_matrix)
        if num ~= 2
            coeff_matrix = -1 * ones(num) + 2 * eye(num);
        else
            coeff_matrix = [1, 1; 1, -1];
        end
    end
    
    coeff_matrix = coeff_matrix * global_coeff;
    
    assert(size(coeff_matrix, 1) == num && size(coeff_matrix, 2) == num);
    assert(det(coeff_matrix) ~= 0);
    
    result = '';
    weight_src_names = cell(1, num);
    for i = 1 : num
        weight_src_names{i} = [prefix, '_weightsrc_', num2str(i)];
        result = cat(2, result, builder.layers.add_weight(weight_src_names{i}, bottom, weight_src_names{i}, dims, opt));
    end
    
    output_weight_names = cell(1, num);
    for i = 1 : num
        output_weight_names{i} = [prefix, '_weighttrans_', num2str(i)];
        result = cat(2, result, builder.layers.add_eltsum(output_weight_names{i}, weight_src_names, output_weight_names{i}, coeff_matrix(i, :)));
    end
end

