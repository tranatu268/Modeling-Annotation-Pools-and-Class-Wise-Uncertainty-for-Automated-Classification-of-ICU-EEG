function fig_pool_shifts(datadir, outdir)
%FIG_POOL_SHIFTS  Learned shifts of the low-vote pools relative to the >=10-vote panel (one column, 2 rows).
%   Gray bars: centred log-ratio of raw vote shares (rater + signal differences mixed).
%   RaTE: mu_s of every fit (5 folds x 3 seeds) as small markers.
%   Pool-heads: mean over fits of the per-segment mean residual, bars = mean input-dependent SD.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S = hms_style();
F = hms_read(datadir, 'pool_shifts_fits.csv');
R = hms_read(datadir, 'pool_shifts_raw.csv');

series = struct( ...
    'method',  {"rate",  "rate",   "pheads", "pheads"}, ...
    'backbone',{"cnn1d", "spec2d", "cnn1d",  "spec2d"}, ...
    'dx',      {-0.27,   -0.09,    0.09,     0.27}, ...
    'marker',  {'o',     'd',      's',      '^'}, ...
    'label',   {"RaTE, 1-D CNN", "RaTE, spectrogram", "Pool-heads, 1-D CNN", "Pool-heads, spectrogram"});

fig = figure('Units', 'inches', 'Position', [0.5 0.5 S.col 4.3], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
titles = ["(a) Pool 1 (3-7 votes)", "(b) Pool 2 (1-2 votes)"];
x = 1:6;
axs = gobjects(1, 2);
for pool = 1:2
    ax = nexttile(t); axs(pool) = ax; hold(ax, 'on');
    raw = zeros(1, 6);
    for k = 1:6
        raw(k) = R.raw_logratio(R.pool == pool & R.cls == S.classes(k));
    end
    hBar = bar(ax, x, raw, 0.8, 'FaceColor', S.light, 'EdgeColor', 'none');
    yline(ax, 0, '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 0.5);
    h = gobjects(1, numel(series));
    for j = 1:numel(series)
        sj = series(j);
        c = S.okabe(j, :);
        sel = F.pool == pool & F.method == sj.method & F.backbone == sj.backbone;
        if sj.method == "rate"
            xs = []; ys = [];
            for k = 1:6
                v = F.shift(sel & F.cls == S.classes(k));
                xs = [xs; repmat(x(k) + sj.dx, numel(v), 1)]; %#ok<AGROW>
                ys = [ys; v]; %#ok<AGROW>
            end
            h(j) = plot(ax, xs, ys, sj.marker, 'LineStyle', 'none', 'MarkerSize', 2.6, ...
                        'MarkerFaceColor', c, 'MarkerEdgeColor', 'none');
        else
            m = zeros(1, 6); sd = zeros(1, 6);
            for k = 1:6
                m(k)  = mean(F.shift(sel & F.cls == S.classes(k)));
                sd(k) = mean(F.input_sd(sel & F.cls == S.classes(k)));
            end
            h(j) = errorbar(ax, x + sj.dx, m, sd, sj.marker, 'Color', c, 'MarkerFaceColor', c, ...
                            'MarkerSize', 3.5, 'CapSize', 2, 'LineWidth', 0.7);
        end
    end
    xlim(ax, [0.4 6.6]);
    xticks(ax, x); xticklabels(ax, S.labels);
    title(ax, titles(pool));
    hms_axes(ax, S);
end
linkaxes(axs, 'y');
xticklabels(axs(1), []);                   % same classes in both rows: labels only under the bottom panel
yl = ylabel(t, 'Shift vs \geq10-vote panel (log-odds)');
yl.FontName = S.font; yl.FontSize = S.fsLabel;
lbl = [[series.label], "Raw vote-share log-ratio"];
lg = hms_legend(axs(1), [h, hBar], lbl, S, 2);
lg.Orientation = 'horizontal';             % fill row by row: RaTE | pool-heads | raw
lg.NumColumns = 2;
hms_export(fig, outdir, 'pool_shifts');
end
