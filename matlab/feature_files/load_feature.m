function [pic_info, output_feature] = load_feature(path, nViews)
    fid = fopen(path, 'r');
    
    feature_dim = fread(fid, 1, 'int');
    
    pic_info = {};
    output_feature = [];
    while 1
        pic_name = read_string(fid);
        if feof(fid)
            break;
        end
        
        cls_name = read_string(fid);
        multiview_feature = fread(fid, [feature_dim, nViews], 'float');
        singleview_feature = sum(multiview_feature, 2);
        output_feature = [output_feature, singleview_feature];
        info.pic_name = pic_name;
        info.cls_name = cls_name;
        pic_info{end + 1} = info;
    end
    
    fclose(fid);
end

