function result = add_matrix_mul(name, bottom, top, trans_a, trans_b)
    trans_a = iif(trans_a, 'true', 'false');
    trans_b = iif(trans_b, 'true', 'false');

    result = builder.layers.add_layer(name, 'matrix_mul', bottom, top, ...
        {'matrix_mul_param', { ...
            'trans_A', trans_a, ...
            'trans_B', trans_b, ...
            } ...
        } ...
        );
end

