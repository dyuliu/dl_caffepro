function result = add_concat(name, bottom, top, concat_dim)
    assert(nargin >= 3);
    if nargin <= 3, concat_dim = 1; end;
    result = builder.layers.add_layer(name, 'concat', bottom, top, 'concat_param', { 'concat_dim', concat_dim });
end

