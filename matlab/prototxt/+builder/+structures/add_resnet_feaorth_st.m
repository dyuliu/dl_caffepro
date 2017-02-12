function [result, output_name] = add_resnet_feaorth_st(prefix, bottom, outputs, stride, has_proj, bn_stat)
    assert(nargin >= 3);
    
    if nargin <= 3, stride = 1; end;
    if nargin <= 4, has_proj = false; end;
    if nargin <= 5, bn_stat = 'NOT_RECORD'; end;
    
    opt.conv_rename_fun = @(pre) ['interstellar', pre];
    opt.bn_rename_fun = @(pre) ['bn', pre];
    opt.relu_rename_fun = @(pre) ['interstellar', pre, '_relu'];
    
    output_name = ['ins', prefix];
    result = ['#-------interstellar ', prefix, '------ (', bottom, '=>', output_name, ')', nl, nl];
    
    path1_output = bottom;
    if has_proj
        result = cat(2, result, '#------1x1(', num2str(outputs), ')', nl, nl);
        [r1, path1_output] = builder.structures.add_conv_st([prefix, '_branch1'], bottom, 1, outputs, stride, 0, false, true, false, bn_stat, opt);
        result = cat(2, result, r1);
    end
    
    result = cat(2, result, '#------1x1-3x3-1x1(', num2str(outputs), ', ', num2str(outputs), ')', nl, nl);
    [r2a, path2a_output] = builder.structures.add_conv_st([prefix, '_branch2a'], bottom, 3, outputs, stride, 1, false, true, true, bn_stat, opt);
    [r2b, path2b_output] = builder.structures.add_conv_st([prefix, '_branch2b'], path2a_output, 3, outputs, 1, 1, false, true, false, bn_stat, opt);
    result = cat(2, result, r2a, r2b);
    
    result = cat(2, result, builder.layers.add_eltsum(['interstellar', prefix], {path1_output, path2b_output}, output_name), nl);
    
    mean_dummy = [prefix, '_mean_dummy'];
    result = cat(2, result, builder.layers.add_dummy(mean_dummy, output_name, mean_dummy));
    mn_name = [prefix, '_mn'];
    result = cat(2, result, builder.layers.add_bn(mn_name, mean_dummy, mn_name, bn_stat, false));
    trans_name = [prefix, '_trans'];
    result = cat(2, result, builder.layers.add_transpose(trans_name, mn_name, trans_name, 3));
    fea_norm_name = [prefix, '_feanorm'];
    result = cat(2, result, builder.layers.add_layer(fea_norm_name, 'l2norm', trans_name, fea_norm_name));
    fea_dummy = [prefix, '_feanorm_dummy'];
    result = cat(2, result, builder.layers.add_dummy(fea_dummy, fea_norm_name, fea_dummy));
    fea_mul_name = [prefix, '_feamul'];
    result = cat(2, result, builder.layers.add_matrix_mul(fea_mul_name, {fea_norm_name, fea_dummy}, fea_mul_name, false, true));
    result = cat(2, result, builder.layers.add_diag_operation([prefix, '_diag'], fea_mul_name, fea_mul_name, 1, -1));
    weight_loss_name = [prefix, '_loss'];
    result = cat(2, result, builder.layers.add_l2loss(weight_loss_name, fea_mul_name, weight_loss_name, 1, false));
    
    result = cat(2, result, builder.layers.add_layer(['ins', prefix, '_relu'], 'relu', output_name, [output_name, '_relu']), nl);
    output_name = [output_name, '_relu'];
end

