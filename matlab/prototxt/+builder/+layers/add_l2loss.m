function result = add_l2loss(name, bottom, top, coeff, display_result)
    assert(nargin >= 3);
    if nargin <= 3, coeff = []; end;
    if nargin <= 4, display_result = true; end;

    result = builder.layers.add_layer( ...
        name, 'euclidean_loss', bottom, top, ...
        {'loss_param', { ...
            'coeff', coeff, ...
            'display_result', iif(display_result, 'true', 'false')
            } ...
        } ...
        );
end
