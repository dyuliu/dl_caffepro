function [result, output_name] = add_InceptionA(prefix, bottom, params, bn_stat, opt)
    assert(nargin >= 3);
    if nargin <= 3, bn_stat = 'NOT_RECORD'; end;
    if nargin <= 4, opt = struct(); end;
    
    assert(length(params) == 7);
    
    result = ['#----------InceptionA ', prefix, '-----------', nl];
    pool_name = [prefix, '_pool'];
    r_pool = builder.layers.add_pool(pool_name, bottom, pool_name, 'AVE', 3, 1, 1);
    [r_branchpool, branchpool_name] = builder.structures.add_conv_st([prefix, '_branchpool'], pool_name, 1, params(1), 1, 0, false, true, true, bn_stat, opt);
    
    [r_branch1x1, branch1x1_name] = builder.structures.add_conv_st([prefix, '_branch1x1'], bottom, 1, params(2), 1, 0, false, true, true, bn_stat, opt);
    
    [r_branch1x1_3x3a, branch1x1_3x3a_name] = builder.structures.add_conv_st([prefix, '_branch1x1_3x3a'], bottom, 1, params(3), 1, 0, false, true, true, bn_stat, opt);
    [r_branch1x1_3x3b, branch1x1_3x3b_name] = builder.structures.add_conv_st([prefix, '_branch1x1_3x3b'], branch1x1_3x3a_name, 3, params(4), 1, 1, false, true, true, bn_stat, opt);
    
    [r_branch1x1_duo3x3a, branch1x1_duo3x3a_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo3x3a'], bottom, 1, params(5), 1, 0, false, true, true, bn_stat, opt);
    [r_branch1x1_duo3x3b, branch1x1_duo3x3b_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo3x3b'], branch1x1_duo3x3a_name, 3, params(6), 1, 1, false, true, true, bn_stat, opt);
    [r_branch1x1_duo3x3c, branch1x1_duo3x3c_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo3x3c'], branch1x1_duo3x3b_name, 3, params(7), 1, 1, false, true, true, bn_stat, opt);
    
    output_name = [prefix, '_concat'];
    r_concat = builder.layers.add_concat(output_name, {branchpool_name, branch1x1_name, branch1x1_3x3b_name, branch1x1_duo3x3c_name}, output_name, 1);
    
    result = cat(2, result, r_pool, r_branchpool, r_branch1x1, r_branch1x1_3x3a, r_branch1x1_3x3b, r_branch1x1_duo3x3a, r_branch1x1_duo3x3b, r_branch1x1_duo3x3c, r_concat);
end

