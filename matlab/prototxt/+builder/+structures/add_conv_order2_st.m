function [result, output_name] = add_conv_order2_st(prefix, bottom, ksize, outputs, stride, pad, has_bias, has_bn, has_relu, bn_stat, varargin)
    assert(nargin >= 9);
    if nargin <= 9, bn_stat = 'NOT_RECORD'; end;
    opt = struct();
    if nargin >= 11, opt = varargin{1}; end;
    
    assert(stride == 1);
    
    conv_name = prefix;
    bn_name = ['bn_', prefix];
    relu_name = [prefix, '_relu'];
    if isfield(opt, 'conv_rename_fun'), conv_name = opt.conv_rename_fun(prefix); end;
    if isfield(opt, 'bn_rename_fun'), bn_name = opt.bn_rename_fun(prefix); end;
    if isfield(opt, 'relu_rename_fun'), relu_name = opt.relu_rename_fun(prefix); end;
    
    conv_linear_name = [conv_name, '_linear'];
    result = [builder.layers.add_conv(conv_linear_name, bottom, conv_linear_name, ksize, outputs - 1, stride, pad, has_bias, opt), nl];
    
    conv_qu_name = [conv_name, '_order2'];
    result = cat(2, result, builder.layers.add_conv(conv_qu_name, bottom, conv_qu_name, 1, outputs, 1, 0, has_bias, opt));
    dot_prod_name = [conv_name, '_order2dot'];
    result = cat(2, result, builder.layers.add_dim_innerproduct(dot_prod_name, {bottom, conv_qu_name}, dot_prod_name, 2));
    
    output_name = conv_name;
    if has_bn, output_name = [conv_name, '_t']; end;
    result = cat(2, result, builder.layers.add_concat(output_name, {conv_linear_name, dot_prod_name}, output_name, 1));
    
    if has_bn 
        result = cat(2, result, builder.layers.add_bn(bn_name, output_name, conv_name, bn_stat), nl);
        output_name = conv_name;
    end
    
    if has_relu
        result = cat(2, result, builder.layers.add_layer(relu_name, 'relu', output_name, conv_name), nl);
        output_name = conv_name;
    end
end

