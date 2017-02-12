function result = add_transpose(name, bottom, top, lead_dim, dims)
    assert(nargin >= 4);
    if nargin <= 4, dims = []; end;
    
    dm = {};
    if ~isempty(dims)
        dm = cell(1, length(dims));
        for i = 1 : length(dims)
            dm{i * 2 - 1} = 'output_dims';
            dm{i * 2} = dims(i);
        end
    end
    
    result = builder.layers.add_layer(name, 'transpose', bottom, top, ...
        {'transpose_param', { ...
            'lead_dim', lead_dim, ...
            dm, ...
            }
        } ...
        );
end

