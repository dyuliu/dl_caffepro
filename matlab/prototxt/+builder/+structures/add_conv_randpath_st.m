function [result, output_name] = add_conv_randpath_st(prefix, bottom, ksize, inputs, outputs, stride, pad, has_bias, has_bn, has_relu, bn_stat, varargin)
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
    
    weight_trans_pool = [prefix, '_weight_transpool'];
    result = cat(2, result, builder.layers.add_pool(weight_trans_pool, weight_src_name, weight_trans_pool, 'AVE', ksize, 1));
    
    result = cat(2, result, [builder.layers.add_correlation(conv_name, {bottom, weight_src_name}, conv_name, stride, pad, has_bias, opt), nl]);
    conv_trans_name = [conv_name, '_trans'];
    result = cat(2, result, [builder.layers.add_correlation(conv_trans_name, {bottom, weight_trans_pool}, conv_trans_name, stride, 0, has_bias, opt), nl]);
    
    path1_name = conv_name;
    path2_name = conv_trans_name;
    
    if has_bn 
        path1_bn_name = [path1_name, '_bn'];
        path2_bn_name = [path2_name, '_bn'];
        result = cat(2, result, builder.layers.add_bn(path1_bn_name, path1_name, path1_bn_name, bn_stat), nl);
        result = cat(2, result, builder.layers.add_bn(path2_bn_name, path2_name, path2_bn_name, bn_stat), nl);
        path1_name = path1_bn_name;
        path2_name = path2_bn_name;
    end
    
    output_name = [prefix, '_output'];
    result = cat(2, result, builder.layers.add_layer(output_name, 'rand_select', {path1_name, path2_name}, output_name));
    
    if has_relu
        result = cat(2, result, builder.layers.add_layer(relu_name, 'relu', output_name, output_name), nl);
    end
end

