function fig_reliability(datadir, outdir)
%FIG_RELIABILITY  Fig. 3: class-wise reliability against the fraction of expert votes (10 equal-count bins).
%   One row of six class panels. Class-wise calibration errors are printed in the empty lower-right corner
%   of each panel (below the diagonal); the legend sits in its own tile under the panels.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S = hms_style();
B  = hms_read(datadir, 'reliability_bins.csv');
CE = hms_read(datadir, 'reliability_ce.csv');
models = struct('method', {"kl", "pheads", "rate"}, 'backbone', {"spec2d", "spec2d", "cnn1d"}, ...
                'marker', {'o', 's', '^'}, 'style', {'-', '-', '--'}, ...
                'label', {"Two-stage, spectrogram CNN", "Pool-heads, spectrogram CNN", "RaTE, 1-D CNN"});

fig = figure('Units', 'inches', 'Position', [0.5 0.5 S.wide 1.95], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig, 1, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:6
    ax = nexttile(t); hold(ax, 'on');
    lim = 0;
    for j = 1:numel(models)
        sel = B.method == models(j).method & B.backbone == models(j).backbone & B.cls == S.classes(k);
        lim = max([lim; B.mean_pred(sel); B.mean_votes(sel)]);
    end
    lim = min(1, 1.08 * lim);
    hDiag = plot(ax, [0 lim], [0 lim], ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.7);
    h = gobjects(1, numel(models));
    for j = 1:numel(models)
        m = models(j); c = S.okabe(j, :);
        sel = B.method == m.method & B.backbone == m.backbone & B.cls == S.classes(k);
        [xp, o] = sort(B.mean_pred(sel)); yv = B.mean_votes(sel); yv = yv(o);
        h(j) = plot(ax, xp, yv, m.style, 'Color', c, 'LineWidth', S.lw, 'Marker', m.marker, ...
                    'MarkerSize', 2.8, 'MarkerFaceColor', c);
        ce = CE.classwise_ce(CE.method == m.method & CE.backbone == m.backbone & CE.cls == S.classes(k));
        text(ax, 0.98, 0.04 + 0.12 * (numel(models) - j), sprintf('%.3f', ce), 'Units', 'normalized', ...
             'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'Color', c, ...
             'FontName', S.font, 'FontSize', S.fsNote);
    end
    axis(ax, [0 lim 0 lim]); pbaspect(ax, [1 1 1]);
    title(ax, S.labels(k));
    hms_axes(ax, S);
    if k == 1, h1 = h; ax1 = ax; end
end
xlabel(t, 'Predicted probability', 'FontName', S.font, 'FontSize', S.fsLabel);
ylabel(t, 'Fraction of expert votes', 'FontName', S.font, 'FontSize', S.fsLabel);
hms_legend(ax1, [h1, hDiag], [[models.label], "Perfect calibration"], S, 4);
hms_export(fig, outdir, 'reliability');
end
