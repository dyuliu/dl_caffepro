function [result, output_name] = add_InceptionC(prefix, bottom, params, bn_stat, opt)
    assert(nargin >= 3);
    if nargin <= 3, bn_stat = 'NOT_RECORD'; end;
    if nargin <= 4, opt = struct(); end;
    
    assert(length(params) == 10);
    
    result = ['#----------InceptionC ', prefix, '-----------', nl];
    pool_name = [prefix, '_pool'];
    r_pool = builder.layers.add_pool(pool_name, bottom, pool_name, 'AVE', 3, 1, 1);
    [r_branchpool, branchpool_name] = builder.structures.add_conv_st([prefix, '_branchpool'], pool_name, 1, params(1), 1, 0, false, true, true, bn_stat, opt);
    
    [r_branch1x1, branch1x1_name] = builder.structures.add_conv_st([prefix, '_branch1x1'], bottom, 1, params(2), 1, 0, false, true, true, bn_stat, opt);
    
    [r_branch1x1_1x3a, branch1x1_1x3a_name] = builder.structures.add_conv_st([prefix, '_branch1x1_1x3a'], bottom, 1, params(3), 1, 0, false, true, true, bn_stat, opt);
    [r_branch1x1_1x3b, branch1x1_1x3b_name] = builder.structures.add_conv_st([prefix, '_branch1x1_1x3b'], branch1x1_1x3a_name, [3, 1], params(4), 1, [1, 0], false, true, true, bn_stat, opt);
    [r_branch1x1_1x3c, branch1x1_1x3c_name] = builder.structures.add_conv_st([prefix, '_branch1x1_1x3c'], branch1x1_1x3a_name, [1, 3], params(5), 1, [0, 1], false, true, true, bn_stat, opt);
    
    [r_branch1x1_duo1x3a, branch1x1_duo1x3a_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x3a'], bottom, 1, params(6), 1, 0, false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x3b, branch1x1_duo1x3b_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x3b'], branch1x1_duo1x3a_name, [3, 1], params(7), 1, [1, 0], false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x3c, branch1x1_duo1x3c_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x3c'], branch1x1_duo1x3b_name, [1, 3], params(8), 1, [0, 1], false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x3d, branch1x1_duo1x3d_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x3d'], branch1x1_duo1x3c_name, [1, 3], params(9), 1, [0, 1], false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x3e, branch1x1_duo1x3e_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x3e'], branch1x1_duo1x3c_name, [3, 1], params(10), 1, [1, 0], false, true, true, bn_stat, opt);
    
    output_name = [prefix, '_concat'];
    r_concat = builder.layers.add_concat(output_name, {branchpool_name, branch1x1_name, branch1x1_1x3b_name, branch1x1_1x3c_name, branch1x1_duo1x3d_name, branch1x1_duo1x3e_name}, output_name, 1);
    
    result = cat(2, result, r_pool, r_branchpool, r_branch1x1, r_branch1x1_1x3a, r_branch1x1_1x3b, r_branch1x1_1x3c, r_branch1x1_duo1x3a, r_branch1x1_duo1x3b, r_branch1x1_duo1x3c, ...
        r_branch1x1_duo1x3d, r_branch1x1_duo1x3e, r_concat);
end

