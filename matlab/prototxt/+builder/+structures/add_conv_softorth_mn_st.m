function [result, output_name] = add_conv_softorth_mn_st(prefix, bottom, ksize, inputs, outputs, stride, pad, has_bias, has_mn, has_relu, bn_stat, varargin)
    assert(nargin >= 10);
    if nargin <= 10, bn_stat = 'NOT_RECORD'; end;
    opt = struct();
    if nargin >= 12, opt = varargin{1}; end;
    
    conv_name = prefix;
    bn_name = ['mn_', prefix];
    relu_name = [prefix, '_relu'];
    if isfield(opt, 'conv_rename_fun'), conv_name = opt.conv_rename_fun(prefix); end;
    if isfield(opt, 'bn_rename_fun'), bn_name = opt.bn_rename_fun(prefix); end;
    if isfield(opt, 'relu_rename_fun'), relu_name = opt.relu_rename_fun(prefix); end;
    
    weight_src_name = [prefix, '_weightsrc'];
    result = builder.layers.add_weight(weight_src_name, bottom, weight_src_name, [ksize, ksize, inputs, outputs], opt);
    weight_norm_name = [prefix, '_weightnorm'];
    result = cat(2, result, builder.layers.add_layer(weight_norm_name, 'l2norm', weight_src_name, weight_norm_name));
    weight_dummy = [prefix, '_weightnorm_dummy'];
    result = cat(2, result, builder.layers.add_dummy(weight_dummy, weight_norm_name, weight_dummy));
    weight_mul_name = [prefix, '_weightmul'];
    result = cat(2, result, builder.layers.add_matrix_mul(weight_mul_name, {weight_norm_name, weight_dummy}, weight_mul_name, false, true));
    result = cat(2, result, builder.layers.add_diag_operation([prefix, '_diag'], weight_mul_name, weight_mul_name, 1, -1));
    weight_loss_name = [prefix, '_loss'];
    result = cat(2, result, builder.layers.add_l2loss(weight_loss_name, weight_mul_name, weight_loss_name, 1, false));
    
    output_name = conv_name;
    if has_mn, output_name = [conv_name, '_t']; end;
    result = cat(2, result, [builder.layers.add_correlation(conv_name, {bottom, weight_norm_name}, output_name, stride, pad, has_bias, opt), nl]);
    
    if has_mn 
        result = cat(2, result, builder.layers.add_mn(bn_name, output_name, conv_name, bn_stat), nl);
        output_name = conv_name;
    end
    
    if has_relu
        result = cat(2, result, builder.layers.add_layer(relu_name, 'relu', output_name, conv_name), nl);
        output_name = conv_name;
    end
end

