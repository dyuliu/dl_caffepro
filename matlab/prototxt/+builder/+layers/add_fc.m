function result = add_fc(name, bottom, top, outputs, has_bias, varargin)
    assert(nargin >= 4);
    opt = struct();
    if nargin < 5, has_bias = true; end;
    if nargin >= 6, opt = varargin{1}; end;
    
    if ~isfield(opt, 'weight_filler'), opt.weight_filler = {'type', 'gaussian', 'std', 0.01}; end;
    if ~isfield(opt, 'bias_filler'), opt.bias_filler = {'type', 'constant', 'value', 0}; end;
    
    result = builder.layers.add_layer( ...
        name, 'inner_product', bottom, top, ...
        {'inner_product_param', { ...
            'num_output', outputs, ...
            'weight_filler', opt.weight_filler, ...
            iif(has_bias, {'bias_filler', opt.bias_filler}, []), ...
            'bias_term', iif(has_bias, [], 'false'), ...
            }, ...
        }, ...
        'blobs_lr', 1, ...
        iif(has_bias, {'blobs_lr', 1}, []) ...
        );
end

