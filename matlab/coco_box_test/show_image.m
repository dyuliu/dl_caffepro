function result = show_image(im, boxes, color, b_show)
%SHOW_IMAGE Show image with boxes
%   im: image
%   boxes: bounding boxes (1 based)
%   color: optional. box colors
%   b_show: optional, whether to show the image

    if nargin < 3, color = [0, 255, 0]; end;
    if nargin < 4, b_show = true; end;
    
    im = draw_img_box(im, boxes, color);
    if b_show, imshow(im); end;
    
    result = im;
end

