function T = hms_read(datadir, file)
%HMS_READ  Read one exported CSV with text columns as strings.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
f = fullfile(datadir, file);
if ~isfile(f)
    error('hms:missing', 'Missing %s. Run  python scripts\\14_export_figdata.py  first.', f);
end
T = readtable(f, 'TextType', 'string');
v = T.Properties.VariableNames;
v(strcmp(v, 'class')) = {'cls'};   % avoid a clash with the built-in class()
T.Properties.VariableNames = v;
end
