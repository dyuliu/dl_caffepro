function [result, output_name] = add_conv_feabn_st(prefix, bottom, ksize, outputs, stride, pad, has_bias, has_bn, has_relu, bn_stat, terms, varargin)
    assert(nargin >= 11);
    opt = struct();
    if nargin >= 12, opt = varargin{1}; end;
    
    conv_name = prefix;
    bn_name = ['bn_', prefix];
    relu_name = [prefix, '_relu'];
    if isfield(opt, 'conv_rename_fun'), conv_name = opt.conv_rename_fun(prefix); end;
    if isfield(opt, 'bn_rename_fun'), bn_name = opt.bn_rename_fun(prefix); end;
    if isfield(opt, 'relu_rename_fun'), relu_name = opt.relu_rename_fun(prefix); end;
    
    output_name = conv_name;
    if has_bn, output_name = [conv_name, '_t']; end;
    result = [builder.layers.add_conv(conv_name, bottom, output_name, ksize, outputs, stride, pad, has_bias, opt), nl];
    
    assert(has_bn);
    bn_dummy = [prefix, '_bndummy'];
    result = cat(2, result, builder.layers.add_dummy(bn_dummy, output_name, bn_dummy));
    result = cat(2, result, builder.layers.add_bn(bn_name, bn_dummy, bn_name, bn_stat), nl);
    
    trans_name = [prefix, '_trans'];
    result = cat(2, result, builder.layers.add_transpose(trans_name, bn_name, trans_name, 3));
    fea_dummy = [prefix, '_trans_dummy'];
    result = cat(2, result, builder.layers.add_dummy(fea_dummy, trans_name, fea_dummy));
    fea_mul_name = [prefix, '_feamul'];
    result = cat(2, result, builder.layers.add_matrix_mul(fea_mul_name, {trans_name, fea_dummy}, fea_mul_name, false, true));
    
    result = cat(2, result, builder.layers.add_diag_operation([prefix, '_diag'], fea_mul_name, fea_mul_name, 1, -1));
    weight_loss_name = [prefix, '_loss'];
    result = cat(2, result, builder.layers.add_l2loss(weight_loss_name, fea_mul_name, weight_loss_name, 1, false));
    
    if has_relu
        result = cat(2, result, builder.layers.add_layer(relu_name, 'relu', output_name, relu_name), nl);
        output_name = relu_name;
    end
end

