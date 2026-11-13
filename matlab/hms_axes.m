function hms_axes(ax, S)
%HMS_AXES  Apply the shared axes style (fonts, ticks, no top/right box).
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
set(ax, 'FontName', S.font, 'FontSize', S.fs, 'LineWidth', S.axlw, 'Box', 'off', ...
    'TickDir', 'out', 'TickLength', [0.02 0.02], 'Layer', 'top', 'Color', 'none');
ax.Title.FontSize = S.fsLabel;
ax.Title.FontWeight = 'normal';
ax.XLabel.FontSize = S.fsLabel;
ax.YLabel.FontSize = S.fsLabel;
end
