function [result, output_name] = add_InceptionB(prefix, bottom, params, bn_stat, opt)
    assert(nargin >= 3);
    if nargin <= 3, bn_stat = 'NOT_RECORD'; end;
    if nargin <= 4, opt = struct(); end;
    
    assert(length(params) == 10);
    
    result = ['#----------InceptionB ', prefix, '-----------', nl];
    pool_name = [prefix, '_pool'];
    r_pool = builder.layers.add_pool(pool_name, bottom, pool_name, 'AVE', 3, 1, 1);
    [r_branchpool, branchpool_name] = builder.structures.add_conv_st([prefix, '_branchpool'], pool_name, 1, params(1), 1, 0, false, true, true, bn_stat, opt);
    
    [r_branch1x1, branch1x1_name] = builder.structures.add_conv_st([prefix, '_branch1x1'], bottom, 1, params(2), 1, 0, false, true, true, bn_stat, opt);
    
    [r_branch1x1_1x7a, branch1x1_1x7a_name] = builder.structures.add_conv_st([prefix, '_branch1x1_1x7a'], bottom, 1, params(3), 1, 0, false, true, true, bn_stat, opt);
    [r_branch1x1_1x7b, branch1x1_1x7b_name] = builder.structures.add_conv_st([prefix, '_branch1x1_1x7b'], branch1x1_1x7a_name, [7, 1], params(4), 1, [3, 0], false, true, true, bn_stat, opt);
    [r_branch1x1_1x7c, branch1x1_1x7c_name] = builder.structures.add_conv_st([prefix, '_branch1x1_1x7c'], branch1x1_1x7b_name, [1, 7], params(5), 1, [0, 3], false, true, true, bn_stat, opt);
    
    [r_branch1x1_duo1x7a, branch1x1_duo1x7a_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7a'], bottom, 1, params(6), 1, 0, false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x7b, branch1x1_duo1x7b_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7b'], branch1x1_duo1x7a_name, [7, 1], params(7), 1, [3, 0], false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x7c, branch1x1_duo1x7c_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7c'], branch1x1_duo1x7b_name, [1, 7], params(8), 1, [0, 3], false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x7d, branch1x1_duo1x7d_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7d'], branch1x1_duo1x7c_name, [7, 1], params(9), 1, [3, 0], false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x7e, branch1x1_duo1x7e_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7e'], branch1x1_duo1x7d_name, [1, 7], params(10), 1, [0, 3], false, true, true, bn_stat, opt);
    
    output_name = [prefix, '_concat'];
    r_concat = builder.layers.add_concat(output_name, {branchpool_name, branch1x1_name, branch1x1_1x7c_name, branch1x1_duo1x7e_name}, output_name, 1);
    
    result = cat(2, result, r_pool, r_branchpool, r_branch1x1, r_branch1x1_1x7a, r_branch1x1_1x7b, r_branch1x1_1x7c, r_branch1x1_duo1x7a, r_branch1x1_duo1x7b, r_branch1x1_duo1x7c, ...
        r_branch1x1_duo1x7d, r_branch1x1_duo1x7e, r_concat);
end

