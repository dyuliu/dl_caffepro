function result = add_eltsum(name, bottom, top, coeff)
    assert(nargin >= 3);
    if nargin <= 3, coeff = []; end;
    
    terms = 1;
    if iscell(bottom), terms = length(bottom); end;
    
    co = {};
    if ~isempty(coeff)
        co = cell(1, terms * 2);
        assert(length(coeff) == terms);
        for i = 1 : terms
            co{i * 2 - 1} = 'coeff';
            co{i * 2} = coeff(i);
        end
    end
    
    result = builder.layers.add_layer( ...
        name, 'eltwise_sum', bottom, top, 'eltwise_sum_param', co ...
        );
end

