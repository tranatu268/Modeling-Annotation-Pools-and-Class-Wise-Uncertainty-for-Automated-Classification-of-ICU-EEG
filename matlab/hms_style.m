function S = hms_style()
%HMS_STYLE  Shared IEEE figure style for the HMS paper figures.
%   Okabe-Ito colours (colour-blind safe), marker shapes and line styles chosen so that every series
%   stays distinguishable in grayscale, and sizes in points for a 7.16-in (two-column) figure.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S.okabe   = [  0 114 178;   % blue
             213  94   0;   % vermilion
               0 158 115;   % bluish green
             204 121 167;   % reddish purple
             230 159   0;   % orange
              86 180 233;   % sky blue
               0   0   0] / 255;
S.gray    = [0.55 0.55 0.55];
S.light   = [0.88 0.88 0.88];
S.font    = 'Helvetica';
S.fs      = 7;      % tick and legend text (pt) at final size
S.fsLabel = 7.5;    % axis labels and panel titles (pt)
S.fsNote  = 6;      % small in-panel numbers (pt)
S.lw      = 1.0;    % data lines (pt)
S.axlw    = 0.5;    % axes lines (pt)
S.wide    = 7.16;   % IEEE two-column width (in)
S.col     = 3.5;    % IEEE one-column width (in)
S.classes = ["seizure", "lpd", "gpd", "lrda", "grda", "other"];
S.labels  = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"];
end
