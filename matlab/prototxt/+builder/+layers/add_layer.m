function result = add_layer(name, type, bottom, top, varargin)
    assert(nargin >= 2);
    if nargin < 3, bottom = {}; end;
    if nargin < 4, top = {}; end;
    
    result = [...
        'layers {', nl, ...
        '  layer {', nl, ...
        '    name: "', name, '"', nl, ...
        '    type: "', type, '"', nl, ...
        ];
    
    result = cat(2, result, builder.layers.add_content(varargin, 4));
    result = cat(2, result, '  }', nl);
    
    if ~isempty(bottom)
        if iscell(bottom)
            for i = 1 : length(bottom)
                result = cat(2, result, builder.layers.add_prop('bottom', bottom{i}, 2));
            end
        else
            result = cat(2, result, builder.layers.add_prop('bottom', bottom, 2));
        end
    end
    
    if ~isempty(top)
        if iscell(top)
            for i = 1 : length(top)
                result = cat(2, result, builder.layers.add_prop('top', top{i}, 2));
            end
        else
            result = cat(2, result, builder.layers.add_prop('top', top, 2));
        end
    end
    
    result = cat(2, result, '}', nl);
end

