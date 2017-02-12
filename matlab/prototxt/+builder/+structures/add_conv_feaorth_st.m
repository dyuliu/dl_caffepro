function [result, output_name] = add_conv_feaorth_st(prefix, bottom, ksize, outputs, stride, pad, has_bias, has_bn, has_relu, bn_stat, varargin)
    assert(nargin >= 9);
    if nargin <= 9, bn_stat = 'NOT_RECORD'; end;
    opt = struct();
    if nargin >= 11, opt = varargin{1}; end;
    
    conv_name = prefix;
    bn_name = ['bn_', prefix];
    relu_name = [prefix, '_relu'];
    if isfield(opt, 'conv_rename_fun'), conv_name = opt.conv_rename_fun(prefix); end;
    if isfield(opt, 'bn_rename_fun'), bn_name = opt.bn_rename_fun(prefix); end;
    if isfield(opt, 'relu_rename_fun'), relu_name = opt.relu_rename_fun(prefix); end;
    
    output_name = conv_name;
    if has_bn, output_name = [conv_name, '_t']; end;
    result = [builder.layers.add_conv(conv_name, bottom, output_name, ksize, outputs, stride, pad, has_bias, opt), nl];
    
    if has_bn 
        result = cat(2, result, builder.layers.add_bn(bn_name, output_name, conv_name, bn_stat), nl);
        output_name = conv_name;
    end
    
    trans_name = [prefix, '_trans'];
    result = cat(2, result, builder.layers.add_transpose(trans_name, output_name, trans_name, 3));
    fea_norm_name = [prefix, '_feanorm'];
    result = cat(2, result, builder.layers.add_layer(fea_norm_name, 'l2norm', trans_name, fea_norm_name));
    fea_dummy = [prefix, '_feanorm_dummy'];
    result = cat(2, result, builder.layers.add_dummy(fea_dummy, fea_norm_name, fea_dummy));
    fea_mul_name = [prefix, '_feamul'];
    result = cat(2, result, builder.layers.add_matrix_mul(fea_mul_name, {fea_norm_name, fea_dummy}, fea_mul_name, false, true));
    result = cat(2, result, builder.layers.add_diag_operation([prefix, '_diag'], fea_mul_name, fea_mul_name, 1, -1));
    weight_loss_name = [prefix, '_loss'];
    result = cat(2, result, builder.layers.add_l2loss(weight_loss_name, fea_mul_name, weight_loss_name, 0.1, false));
    
    if has_relu
        result = cat(2, result, builder.layers.add_layer(relu_name, 'relu', output_name, relu_name), nl);
        output_name = relu_name;
    end
end

