function result = add_prop(name, value, indent)
    if isempty(value) && ~ischar(value)
        result = '';
        return;
    end

    result = [repmat(' ', 1, indent), name, ': '];
    if ischar(value)
        if strcmp(value, 'true') || strcmp(value, 'false')
            result = cat(2, result, value);
        else
            result = cat(2, result, '"', value, '"');
        end
    elseif isstruct(value)
        result = cat(2, result, value.value);
    else
        result = cat(2, result, num2str(value));
    end
    result = cat(2, result, nl);
end

