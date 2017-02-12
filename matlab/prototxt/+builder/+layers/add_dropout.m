function result = add_dropout(name, bottom, top, drop_ratio)
    result = builder.layers.add_layer(name, 'dropout', bottom, top, 'dropout_param', { 'dropout_ratio', drop_ratio });
end

