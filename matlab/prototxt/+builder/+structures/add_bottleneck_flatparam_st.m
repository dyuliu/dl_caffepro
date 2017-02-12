function [result, output_name] = add_bottleneck_flatparam_st(prefix, bottom, bottleneck_outputs, inputs, outputs, stride, has_proj, bn_stat)
    assert(nargin >= 5);
    
    if nargin <= 5, stride = 1; end;
    if nargin <= 6, has_proj = false; end;
    if nargin <= 7, bn_stat = 'NOT_RECORD'; end;
    
    assert(stride == 1)
    
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
    
    result = cat(2, result, '#------1x1-3x3-1x1(', num2str(bottleneck_outputs), ', ', num2str(outputs), ')', nl, nl);
    
    weight_name1x1a = [prefix, '_1x1aweight'];
    rw_1x1asrc = builder.layers.add_weight(weight_name1x1a, bottom, weight_name1x1a, [1, 1, inputs, bottleneck_outputs], opt);
    [r2a, path2a_output] = builder.structures.add_correlation_st([prefix, '_branch2a'], bottom, weight_name1x1a, stride, 0, false, true, true, bn_stat, opt);
    
    weight_name3x3 = [prefix, '_3x3weight'];
    weightsrc_name3x3 = [weight_name3x3, '_src'];
    rw_3x3src = builder.layers.add_weight(weightsrc_name3x3, bottom, weightsrc_name3x3, [3, 3, inputs, bottleneck_outputs], opt);
    [r2b_wa, path2b_wa_output] = builder.structures.add_correlation_st([prefix, '_branch2b_wa'], weightsrc_name3x3, weight_name1x1a, stride, 0, false, true, true, bn_stat, opt);
    [r2b, path2b_output] = builder.structures.add_correlation_st([prefix, '_branch2b'], path2a_output, path2b_wa_output, 1, 1, false, true, true, bn_stat, opt);
    
    weight_name1x1b = [prefix, '_1x1bweight'];
    weightsrc_name1x1b = [weight_name1x1b, '_src'];
    rw_1x1bsrc = builder.layers.add_weight(weightsrc_name1x1b, bottom, weightsrc_name1x1b, [3, 3, inputs, outputs], opt);
    [r2c_wa, path2c_wa_output] = builder.structures.add_correlation_st([prefix, '_branch2c_wa'], weightsrc_name1x1b, weight_name1x1a, stride, 0, false, true, true, bn_stat, opt);
    [r2c_wb, path2c_wb_output] = builder.structures.add_correlation_st([prefix, '_branch2c_wb'], path2c_wa_output, path2b_wa_output, 1, 0, false, true, true, bn_stat, opt);
    [r2c, path2c_output] = builder.structures.add_correlation_st([prefix, '_branch2c'], path2b_output, path2c_wb_output, 1, 0, false, true, false, bn_stat, opt);
    
    result = cat(2, result, rw_1x1asrc, r2a, rw_3x3src, r2b_wa, r2b, rw_1x1bsrc, r2c_wa, r2c_wb, r2c);
    
    result = cat(2, result, builder.layers.add_eltsum(['interstellar', prefix], {path1_output, path2c_output}, output_name, [1, 0.1]), nl);
    result = cat(2, result, builder.layers.add_layer(['ins', prefix, '_relu'], 'relu', output_name, output_name), nl);
end
