function [result, output_name] = add_bottleneck_ldropout_p2_st(prefix, bottom, bottleneck_outputs, outputs, init_value, stride, has_proj, bn_stat)
    assert(nargin >= 5);
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
        [r1, path1_output] = builder.structures.add_conv_st([prefix, '_branch1'], bottom, 1, outputs, stride, 0, false, true, false, bn_stat, opt);
        result = cat(2, result, r1);
    end
    
    part_bottleneck_outputs = round(max(roots([18, 4 * outputs, -2 * outputs * bottleneck_outputs - 9 * bottleneck_outputs .^ 2])));
    
    result = cat(2, result, '#------1x1-3x3-1x1(', num2str(bottleneck_outputs), ', ', num2str(outputs), ')', nl, nl);
    
    % path 2 p1
    [r2a, path2a_output] = builder.structures.add_conv_st([prefix, '_branch2a_p1'], bottom, 1, part_bottleneck_outputs, stride, 0, false, true, true, bn_stat, opt);
    [r2b, path2b_output] = builder.structures.add_conv_st([prefix, '_branch2b_p1'], path2a_output, 3, part_bottleneck_outputs, 1, 1, false, true, true, bn_stat, opt);
    [r2c, path2c_output] = builder.structures.add_conv_st([prefix, '_branch2c_p1'], path2b_output, 1, outputs, 1, 0, false, true, false, bn_stat, opt);
    result = cat(2, result, r2a, r2b, r2c);
    dropout_name = ['dropout1_', prefix];
    result = cat(2, result, builder.layers.add_learnable_dropout(dropout_name, path2c_output, dropout_name, init_value));
    path2_output1 = dropout_name;
    
    % path 2 p2
    [r2a, path2a_output] = builder.structures.add_conv_st([prefix, '_branch2a_p2'], bottom, 1, part_bottleneck_outputs, stride, 0, false, true, true, bn_stat, opt);
    [r2b, path2b_output] = builder.structures.add_conv_st([prefix, '_branch2b_p2'], path2a_output, 3, part_bottleneck_outputs, 1, 1, false, true, true, bn_stat, opt);
    [r2c, path2c_output] = builder.structures.add_conv_st([prefix, '_branch2c_p2'], path2b_output, 1, outputs, 1, 0, false, true, false, bn_stat, opt);
    result = cat(2, result, r2a, r2b, r2c);
    dropout_name = ['dropout2_', prefix];
    result = cat(2, result, builder.layers.add_learnable_dropout(dropout_name, path2c_output, dropout_name, init_value));
    path2_output2 = dropout_name;
    
    result = cat(2, result, builder.layers.add_eltsum(['interstellar', prefix], {path1_output, path2_output1, path2_output2}, output_name), nl);
    result = cat(2, result, builder.layers.add_layer(['ins', prefix, '_relu'], 'relu', output_name, output_name), nl);
end

