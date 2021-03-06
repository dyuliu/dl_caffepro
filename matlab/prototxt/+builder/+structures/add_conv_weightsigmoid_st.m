function [result, output_name] = add_conv_weightsigmoid_st(prefix, bottom, ksize, inputs, outputs, stride, pad, has_bias, has_bn, has_relu, bn_stat, varargin)
    assert(nargin >= 10);
    if nargin <= 10, bn_stat = 'NOT_RECORD'; end;
    opt = struct();
    if nargin >= 12, opt = varargin{1}; end;
    
    conv_name = prefix;
    bn_name = ['bn_', prefix];
    relu_name = [prefix, '_relu'];
    if isfield(opt, 'conv_rename_fun'), conv_name = opt.conv_rename_fun(prefix); end;
    if isfield(opt, 'bn_rename_fun'), bn_name = opt.bn_rename_fun(prefix); end;
    if isfield(opt, 'relu_rename_fun'), relu_name = opt.relu_rename_fun(prefix); end;
    
    weight_src_name = [prefix, '_weightsrc'];
    result = builder.layers.add_weight(weight_src_name, bottom, weight_src_name, [ksize, ksize, inputs, outputs], opt);
    result = cat(2, result, builder.layers.add_layer([prefix, '_sigmoid'], 'sym_sigmoid', weight_src_name, weight_src_name));
    
    output_name = conv_name;
    if has_bn, output_name = [conv_name, '_t']; end;
    result = cat(2, result, [builder.layers.add_correlation(conv_name, {bottom, weight_src_name}, output_name, stride, pad, has_bias, opt), nl]);
    
    if has_bn 
        result = cat(2, result, builder.layers.add_bn(bn_name, output_name, conv_name, bn_stat), nl);
        output_name = conv_name;
    end
    
    if has_relu
        result = cat(2, result, builder.layers.add_layer(relu_name, 'relu', output_name, conv_name), nl);
        output_name = conv_name;
    end
end

