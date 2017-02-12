function [result, output_name, weight_output_name] = add_resweight_bottleneck_st(prefix, bottom, weight_bottom, bottleneck_outputs, outputs, bn_stat, opt)
    assert(nargin >= 5);
    if nargin <= 5, bn_stat = 'NOT_RECORD'; end;
    if nargin <= 6, opt = struct(); end;
    
    assert(isempty(weight_bottom) || length(weight_bottom) == 3);
    
    if ~isfield(opt, 'conv_rename_fun'), opt.conv_rename_fun = @(pre) ['interstellar', pre]; end;
    if ~isfield(opt, 'bn_rename_fun'), opt.bn_rename_fun = @(pre) ['bn', pre]; end;
    if ~isfield(opt, 'relu_rename_fun'), opt.relu_rename_fun = @(pre) ['interstellar', pre, '_relu']; end;
    if ~isfield(opt, 'weight_filler'), opt.weight_filler = { 'type', 'xiangyu' }; end;
    
    eltsum_coeff = [];
    if isfield(opt, 'eltsum_coeff'), eltsum_coeff = opt.eltsum_coeff; end;
    
    output_name = ['ins', prefix];
    result = ['#-------interstellar ', prefix, '------ (', bottom, '=>', output_name, ')', nl, nl];
    
    path1_output = bottom;
    result = cat(2, result, '#------1x1-3x3-1x1(', num2str(bottleneck_outputs), ', ', num2str(outputs), ')', nl, nl);
    if isempty(weight_bottom)
        result = cat(2, result, builder.layers.add_weight([prefix, '_weight2a'], bottom, [prefix, '_weight2a'], [1, 1, outputs, bottleneck_outputs], opt));
        result = cat(2, result, builder.layers.add_weight([prefix, '_weight2b'], bottom, [prefix, '_weight2b'], [3, 3, bottleneck_outputs, bottleneck_outputs], opt));
        result = cat(2, result, builder.layers.add_weight([prefix, '_weight2c'], bottom, [prefix, '_weight2c'], [1, 1, bottleneck_outputs, outputs], opt));
    else
        result = cat(2, result, builder.layers.add_weight([prefix, '_weight2a'], bottom, [prefix, '_weight2a_t'], [1, 1, outputs, bottleneck_outputs], opt));
        result = cat(2, result, builder.layers.add_weight([prefix, '_weight2b'], bottom, [prefix, '_weight2b_t'], [3, 3, bottleneck_outputs, bottleneck_outputs], opt));
        result = cat(2, result, builder.layers.add_weight([prefix, '_weight2c'], bottom, [prefix, '_weight2c_t'], [1, 1, bottleneck_outputs, outputs], opt));
        result = cat(2, result, builder.layers.add_eltsum([prefix, '_w2a_sum'], {weight_bottom{1}, [prefix, '_weight2a_t']}, [prefix, '_weight2a'], eltsum_coeff));
        result = cat(2, result, builder.layers.add_eltsum([prefix, '_w2b_sum'], {weight_bottom{2}, [prefix, '_weight2b_t']}, [prefix, '_weight2b'], eltsum_coeff));
        result = cat(2, result, builder.layers.add_eltsum([prefix, '_w2c_sum'], {weight_bottom{3}, [prefix, '_weight2c_t']}, [prefix, '_weight2c'], eltsum_coeff));
    end
    
    weight_output_name = {[prefix, '_weight2a'], [prefix, '_weight2b'], [prefix, '_weight2c']};
    
    [r2a, path2a_output] = builder.structures.add_correlation_st([prefix, '_branch2a'], bottom, [prefix, '_weight2a'], 1, 0, false, true, true, bn_stat, opt);
    [r2b, path2b_output] = builder.structures.add_correlation_st([prefix, '_branch2b'], path2a_output, [prefix, '_weight2b'], 1, 1, false, true, true, bn_stat, opt);
    [r2c, path2c_output] = builder.structures.add_correlation_st([prefix, '_branch2c'], path2b_output, [prefix, '_weight2c'], 1, 0, false, true, false, bn_stat, opt);
    
    result = cat(2, result, r2a, r2b, r2c);
    
    result = cat(2, result, builder.layers.add_eltsum(['interstellar', prefix], {path1_output, path2c_output}, output_name), nl);
    result = cat(2, result, builder.layers.add_layer(['ins', prefix, '_relu'], 'relu', output_name, output_name), nl);
end

