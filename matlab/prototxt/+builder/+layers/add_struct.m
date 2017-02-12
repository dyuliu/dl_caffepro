function result = add_struct(name, content, indent)
    if isempty(content) 
        result = '';
        return;
    end
    
    result = [repmat(' ', 1, indent), name, ' {', nl];
    result = cat(2, result, builder.layers.add_content(content, indent + 2));
    result = cat(2, result, [repmat(' ', 1, indent), '}', nl]);
end

