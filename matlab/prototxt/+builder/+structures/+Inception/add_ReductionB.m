function [result, output_name] = add_ReductionB(prefix, bottom, params, bn_stat, opt)
    assert(nargin >= 3);
    if nargin <= 3, bn_stat = 'NOT_RECORD'; end;
    if nargin <= 4, opt = struct(); end;
    
    assert(length(params) == 6);
    
    result = ['#----------ReductionB ', prefix, '-----------', nl];
    pool_name = [prefix, '_pool'];
    r_pool = builder.layers.add_pool(pool_name, bottom, pool_name, 'MAX', 3, 2, 0);
    
    [r_branch1x1_3x3a, branch1x1_3x3a_name] = builder.structures.add_conv_st([prefix, '_branch1x1_3x3a'], bottom, 1, params(1), 1, 0, false, true, true, bn_stat, opt);
    [r_branch1x1_3x3b, branch1x1_3x3b_name] = builder.structures.add_conv_st([prefix, '_branch1x1_3x3b'], branch1x1_3x3a_name, 3, params(2), 2, 0, false, true, true, bn_stat, opt);
    
    [r_branch1x1_duo1x7_3x3a, branch1x1_duo1x7_3x3a_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7_3x3a'], bottom, 1, params(3), 1, 0, false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x7_3x3b, branch1x1_duo1x7_3x3b_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7_3x3b'], branch1x1_duo1x7_3x3a_name, [7, 1], params(4), 1, [3, 0], false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x7_3x3c, branch1x1_duo1x7_3x3c_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7_3x3c'], branch1x1_duo1x7_3x3b_name, [1, 7], params(5), 1, [0, 3], false, true, true, bn_stat, opt);
    [r_branch1x1_duo1x7_3x3d, branch1x1_duo1x7_3x3d_name] = builder.structures.add_conv_st([prefix, '_branch1x1_duo1x7_3x3d'], branch1x1_duo1x7_3x3c_name, 3, params(6), 2, 0, false, true, true, bn_stat, opt);
    
    output_name = [prefix, '_concat'];
    r_concat = builder.layers.add_concat(output_name, {pool_name, branch1x1_3x3b_name, branch1x1_duo1x7_3x3d_name}, output_name, 1);
    
    result = cat(2, result, r_pool, r_branch1x1_3x3a, r_branch1x1_3x3b, r_branch1x1_duo1x7_3x3a, r_branch1x1_duo1x7_3x3b, r_branch1x1_duo1x7_3x3c, r_branch1x1_duo1x7_3x3d, r_concat);
end

