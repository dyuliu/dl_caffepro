function y = iif(b_cond, a, b)
%IIF if b_cond == true, return a, otherwise b
    if b_cond
        y = a;
    else
        y = b;
    end
end

