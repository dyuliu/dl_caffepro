function result = add_content(content, indent)
    result = '';
    i = 1;
    while i <= length(content)
        if ~isempty(content{i})
            if iscell(content{i}) 
                result = cat(2, result, builder.layers.add_content(content{i}, indent));
            else 
                assert(i + 1 <= length(content));
                if iscell(content{i + 1})
                    result = cat(2, result, builder.layers.add_struct(content{i}, content{i + 1}, indent));
                else 
                    result = cat(2, result, builder.layers.add_prop(content{i}, content{i + 1}, indent));
                end
                i = i + 1;
            end
        end
        i = i + 1;
    end
end

