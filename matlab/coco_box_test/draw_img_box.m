function result = draw_img_box(im, boxes, color)
    dim = size(boxes);
    assert(length(dim) == 2);
    n = dim(1);
    assert(dim(2) == 4);
    
    result = im;
    
    for i = 1 : n
        box = boxes(i, :);
        if box(1) <= box(3) && box(2) <= box(4) 
            for j = 1 : 3
                result(box(2) : box(4), box(1), j) = color(j);
                result(box(2), box(1) : box(3), j) = color(j);
                result(box(2) : box(4), box(3), j) = color(j);
                result(box(4), box(1) : box(3), j) = color(j);
            end
        end
    end
end

