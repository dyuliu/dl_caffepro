function result = add_householder(name, bottom, top, source)
    assert(nargin >= 3);
    if nargin <= 3, source = 0; end;
    
    result = builder.layers.add_layer(name, 'householder', bottom, top, ...
        {'householder_param', { ...
            'source', source
            }
        } ...
        );
end

