function result = add_dim_innerproduct(name, bottom, top, dim)
    result = builder.layers.add_layer(name, 'dim_innerproduct', bottom, top, ...
        {'dim_innerproduct_param', { ...
            'dim', dim, ...
            }
        } ...
        );
end

