function [result, output_name] = add_bottleneck_weightsigmoid_st(prefix, bottom, bottleneck_outputs, inputs, outputs, stride, has_proj, bn_stat)
    assert(nargin >= 4);
    
    if nargin <= 5, stride = 1; end;
    if nargin <= 6, has_proj = false; end;
    if nargin <= 7, bn_stat = 'NOT_RECORD'; end;
    
    opt.conv_rename_fun = @(pre) ['interstellar', pre];
    opt.bn_rename_fun = @(pre) ['bn', pre];
    opt.relu_rename_fun = @(pre) ['interstellar', pre, '_relu'];
    opt.weight_filler = { 'type', 'xiangyu' };
    
    output_name = ['ins', prefix];
    result = ['#-------interstellar ', prefix, '------ (', bottom, '=>', output_name, ')', nl, nl];
    
    path1_output = bottom;
    if has_proj
        result = cat(2, result, '#------1x1(', num2str(outputs), ')', nl, nl);
        [r1, path1_output] = builder.structures.add_conv_weightsigmoid_st([prefix, '_branch1'], bottom, 1, inputs, outputs, stride, 0, false, true, false, bn_stat, opt);
        result = cat(2, result, r1);
    end
    
    result = cat(2, result, '#------1x1-3x3-1x1(', num2str(bottleneck_outputs), ', ', num2str(outputs), ')', nl, nl);
    [r2a, path2a_output] = builder.structures.add_conv_weightsigmoid_st([prefix, '_branch2a'], bottom, 1, inputs, bottleneck_outputs, stride, 0, false, true, true, bn_stat, opt);
    [r2b, path2b_output] = builder.structures.add_conv_weightsigmoid_st([prefix, '_branch2b'], path2a_output, 3, bottleneck_outputs, bottleneck_outputs, 1, 1, false, true, true, bn_stat, opt);
    [r2c, path2c_output] = builder.structures.add_conv_weightsigmoid_st([prefix, '_branch2c'], path2b_output, 1, bottleneck_outputs, outputs, 1, 0, false, true, false, bn_stat, opt);
    result = cat(2, result, r2a, r2b, r2c);
    
    result = cat(2, result, builder.layers.add_eltsum(['interstellar', prefix], {path1_output, path2c_output}, output_name), nl);
    result = cat(2, result, builder.layers.add_layer(['ins', prefix, '_relu'], 'relu', output_name, output_name), nl);
end

