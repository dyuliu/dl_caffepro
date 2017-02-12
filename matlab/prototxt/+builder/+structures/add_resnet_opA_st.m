function [result, output_name] = add_resnet_opA_st(prefix, bottom, outputs, stride, bn_stat)
    assert(nargin >= 3);
    
    if nargin <= 3, stride = 1; end;
    if nargin <= 4, bn_stat = 'NOT_RECORD'; end;
    
    assert(mod(outputs, 2) == 0);
    
    opt.conv_rename_fun = @(pre) ['interstellar', pre];
    opt.bn_rename_fun = @(pre) ['bn', pre];
    opt.relu_rename_fun = @(pre) ['interstellar', pre, '_relu'];
    
    output_name = ['ins', prefix];
    result = ['#-------interstellar ', prefix, '------ (', bottom, '=>', output_name, ')', nl, nl];
    
    path1_output = [prefix, '_branch1'];
    result = cat(2, result, builder.layers.add_pool(path1_output, bottom, path1_output, 'MAX', 1, stride, [], true));
    
    result = cat(2, result, '#------1x1-3x3-1x1(', num2str(outputs), ', ', num2str(outputs), ')', nl, nl);
    [r2a, path2a_output] = builder.structures.add_conv_st([prefix, '_branch2a'], bottom, 3, outputs, stride, 1, false, true, true, bn_stat, opt);
    [r2b, path2b_output] = builder.structures.add_conv_st([prefix, '_branch2b'], path2a_output, 3, outputs / 2, 1, 1, false, true, false, bn_stat, opt);
    result = cat(2, result, r2a, r2b);
    
    sum_name = ['interstellar', prefix];
    result = cat(2, result, builder.layers.add_eltsum(sum_name, {path1_output, path2b_output}, sum_name), nl);
    
    [r2c, path2c_output] = builder.structures.add_conv_st([prefix, '_branch2c'], path2a_output, 3, outputs / 2, 1, 1, false, true, false, bn_stat, opt);
    result = cat(2, result, r2c);
    
    result = cat(2, result, builder.layers.add_concat(output_name, {sum_name, path2c_output}, output_name, 1));
    result = cat(2, result, builder.layers.add_layer(['ins', prefix, '_relu'], 'relu', output_name, output_name), nl);
end

