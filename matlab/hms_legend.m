function lg = hms_legend(ax, h, labels, S, ncol, where)
%HMS_LEGEND  Legend placed in its own tile of the tiled layout ('south' by default), i.e. outside every
%   axes, so it can never cover data. Requires MATLAB R2020b or later.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
if nargin < 6, where = 'south'; end
lg = legend(ax, h, labels, 'NumColumns', ncol, 'Box', 'off', 'FontName', S.font, ...
            'FontSize', S.fs, 'AutoUpdate', 'off');
lg.ItemTokenSize = [14 8];
lg.Layout.Tile = where;
end
