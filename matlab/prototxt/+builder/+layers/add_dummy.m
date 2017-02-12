function result = add_dummy(name, bottom, top)
    result = builder.layers.add_pool(name, bottom, top, 'AVE', 1, 1, 0);
end

