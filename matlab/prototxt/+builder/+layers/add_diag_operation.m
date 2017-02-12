function result = add_diag_operation(name, bottom, top, scale, shift)
    result = builder.layers.add_layer( ...
        name, 'diag_operation', bottom, top, ...
        {'diag_operation_param', { ...
            'scale', scale, ...
            'shift', shift
            } ...
        } ...
        );
end
