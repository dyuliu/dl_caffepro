function result = add_learnable_dropout(name, bottom, top, init_value)
    result = builder.layers.add_layer(name, 'learnable_dropout', bottom, top, 'learnable_dropout_param', { 'init_value', init_value }, 'blobs_lr', 0);
end

