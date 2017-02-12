function result = add_softmaxloss(name, bottom, top, coeff)
    assert(nargin >= 3);
    if nargin <= 3, coeff = []; end;
    result = builder.layers.add_layer(name, 'softmax_loss', bottom, top, ...
        'loss_param', iif(isempty(coeff), [], { 'coeff', coeff }) ...
        );
end

