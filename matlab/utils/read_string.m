function outputStr = read_string(fid)
%read string from a file
    len = fread(fid, 1, 'int');
    outputStr = fread(fid, [1, len], 'char=>char');
end

