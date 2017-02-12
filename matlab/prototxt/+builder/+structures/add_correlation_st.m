function [result, output_name] = add_correlation_st(prefix, bottom, weight_bottom, stride, pad, has_bias, has_bn, has_relu, bn_stat, varargin)
    assert(nargin >= 8);
    if nargin <= 8, bn_stat = 'NOT_RECORD'; end;
    opt = struct();
    if nargin >= 10, opt = varargin{1}; end;
    
    corr_name = prefix;
    bn_name = ['bn_', prefix];
    relu_name = [prefix, '_relu'];
    if isfield(opt, 'conv_rename_fun'), corr_name = opt.conv_rename_fun(prefix); end;
    if isfield(opt, 'bn_rename_fun'), bn_name = opt.bn_rename_fun(prefix); end;
    if isfield(opt, 'relu_rename_fun'), relu_name = opt.relu_rename_fun(prefix); end;
    
    output_name = corr_name;
    if has_bn, output_name = [corr_name, '_t']; end;
    result = [builder.layers.add_correlation(corr_name, {bottom, weight_bottom}, output_name, stride, pad, has_bias, opt), nl];
    
    if has_bn 
        result = cat(2, result, builder.layers.add_bn(bn_name, output_name, corr_name, bn_stat), nl);
        output_name = corr_name;
    end
    
    if has_relu
        result = cat(2, result, builder.layers.add_layer(relu_name, 'relu', output_name, corr_name), nl);
        output_name = corr_name;
    end
end

