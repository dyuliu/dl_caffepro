function [result, output_name] = add_conv_addi_st(prefix, bottom, ksize, outputs, stride, pad, has_bias, has_bn, has_relu, bn_stat, varargin)
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
    
    sum_name = [prefix, '_sum'];
    result = cat(2, result, builder.layers.add_eltsum(sum_name, {bottom, output_name}, sum_name));
    output_name = sum_name;
    
    if has_bn 
        result = cat(2, result, builder.layers.add_bn(bn_name, output_name, conv_name, bn_stat), nl);
        output_name = conv_name;
    end
    
    if has_relu
        result = cat(2, result, builder.layers.add_layer(relu_name, 'relu', output_name, conv_name), nl);
        output_name = conv_name;
    end
end

