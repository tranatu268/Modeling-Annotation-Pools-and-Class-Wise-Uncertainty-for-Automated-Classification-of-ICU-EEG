%MAKE_ALL_FIGURES  Draw the data figures of the HMS manuscript from work/figdata/*.csv.
%   1) In the repo root (Python):  python scripts\14_export_figdata.py
%   2) In MATLAB (R2020b or later), from this folder:  make_all_figures
%   Output: paper/figs/{pool_shifts,reliability,noise_floor,conformal,roc_experts,eeg_examples,confusion}.pdf (vector) + .png (600 dpi preview).
%   Fig. 1 (method overview) is a diagram drawn with TikZ inside main.tex.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
here    = fileparts(mfilename('fullpath'));
datadir = fullfile(here, '..', 'work', 'figdata');
outdir  = fullfile(here, '..', 'paper', 'figs');
if ~isfolder(outdir), mkdir(outdir); end
if verLessThan('matlab', '9.9')
    error('hms:version', 'MATLAB R2020b or later is needed (legend tiles of tiledlayout, exportgraphics).');
end
fprintf('Fig. 2 pool shifts\n');  fig_pool_shifts(datadir, outdir);
fprintf('Fig. 3 reliability\n');  fig_reliability(datadir, outdir);
fprintf('Fig. 4 conformal\n');    fig_conformal(datadir, outdir);
fprintf('Fig. ROC + PR vs experts\n'); fig_roc_experts(datadir, outdir);
fprintf('Fig. noise floor\n');      fig_noise_floor(datadir, outdir);
fprintf('Fig. EEG examples\n');      fig_eeg_examples(datadir, outdir);
fprintf('Fig. confusion\n');         fig_confusion(datadir, outdir);
fprintf('Done. Check the PNG previews: no legend may overlap a data region.\n');
