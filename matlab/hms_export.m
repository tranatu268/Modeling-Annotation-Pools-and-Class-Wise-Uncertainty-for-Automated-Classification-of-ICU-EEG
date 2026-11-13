function hms_export(fig, outdir, name)
%HMS_EXPORT  Vector PDF for the manuscript and a 600-dpi PNG preview.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
drawnow;
exportgraphics(fig, fullfile(outdir, [name '.pdf']), 'ContentType', 'vector', 'BackgroundColor', 'white');
exportgraphics(fig, fullfile(outdir, [name '.png']), 'Resolution', 600, 'BackgroundColor', 'white');
fprintf('  -> %s.pdf / .png\n', fullfile(outdir, name));
end
