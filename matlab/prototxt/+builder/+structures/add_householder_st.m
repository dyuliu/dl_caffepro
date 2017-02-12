function [result, output_weight_name] = add_householder_st(prefix, bottom, dims, opt, wc)
    assert(nargin >= 3);
    if nargin <= 3, opt = []; end;
    if nargin <= 4, wc = []; end;
    
    weight_src_dims = dims;
    weight_src_dims(end) = 1;
    weight_src_name = [prefix, '_weightsrc'];
    result = builder.layers.add_weight(weight_src_name, bottom, weight_src_name, weight_src_dims, opt, wc);
    
    norm_name = [prefix, '_weightsrc_norm'];
    result = cat(2, result, builder.layers.add_layer(norm_name, 'l2norm', weight_src_name, norm_name));
    
    householder_name = [prefix, '_householder'];
    result = cat(2, result, builder.layers.add_householder(householder_name, norm_name, householder_name, 0));
    
    output_weight_name = [prefix, '_weights'];
    result = cat(2, result, builder.layers.add_instance_sample(output_weight_name, householder_name, output_weight_name, dims(end), 'RAND'));
end

