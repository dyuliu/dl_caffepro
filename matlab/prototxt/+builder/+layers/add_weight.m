function result = add_weight(name, bottom, top, dims, varargin)
    assert(nargin >= 4);
    opt = struct();
    if nargin >= 5, opt = varargin{1}; end;
    wc = [];
    if nargin >= 6, wc = varargin{2}; end;
    
    if ~isfield(opt, 'weight_filler'), opt.weight_filler = {'type', 'gaussian', 'std', 0.01}; end;
    
    assert(~isempty(dims));
    dm = cell(1, length(dims) * 2);
    for i = 1 : length(dims)
        dm{i * 2 - 1} = 'weight_dim';
        dm{i * 2} = dims(i);
    end
    
    result = builder.layers.add_layer(name, 'weight', bottom, top, ...
        {'weight_param', { ...
            dm, ...
            'weight_filler', opt.weight_filler, ...
            }
        }, ...
        'weight_decay', wc ...
        );
end

