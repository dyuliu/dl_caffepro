function [result, output_weight_name] = add_householder_higherorder_nosample_st(prefix, bottom, size, order, opt, wc)
    assert(nargin >= 4);
    if nargin <= 4, opt = []; end;
    if nargin <= 5, wc = []; end;
    
    assert(order >= 1);
    
    weight_src_dims = size;
    weight_src_dims(end) = order;
    weight_src_name = [prefix, '_weightsrc'];
    result = builder.layers.add_weight(weight_src_name, bottom, weight_src_name, weight_src_dims, opt, wc);
    
    norm_name = [prefix, '_weightsrc_norm'];
    result = cat(2, result, builder.layers.add_layer(norm_name, 'l2norm', weight_src_name, norm_name));
    
    householder_name = [prefix, '_householder'];
    result = cat(2, result, builder.layers.add_householder(householder_name, norm_name, householder_name, 0));
    
    for i = 2 : order 
        prev_householder_name = householder_name;
        householder_name = [prefix, '_householder_order', num2str(i)];
        result = cat(2, result, builder.layers.add_householder(householder_name, norm_name, [householder_name, '_t'], i - 1));
        result = cat(2, result, builder.layers.add_matrix_mul([householder_name, '_mul'], {[householder_name, '_t'], prev_householder_name}, householder_name, false, false));
    end
    
    output_weight_name = householder_name;
end

