function fig_conformal(datadir, outdir)
%FIG_CONFORMAL  Prediction sets of the spectrogram pool-head model versus nominal coverage 1-alpha
%   (one column, 2 rows). Line style = calibration rule in both panels: dashed marginal, solid label-conditional.
%   (a) coverage of expert votes per class (colour = class);
%   (b) deferral rate (left axis, filled circles) and mean set size (right axis, open squares).
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S = hms_style();
C = hms_read(datadir, 'conformal.csv');
C = C(C.model == "pheads_spec2d", :);
mar = sortrows(C(C.mode == "marginal", :), 'alpha', 'descend');
lab = sortrows(C(C.mode == "label", :), 'alpha', 'descend');
nom = 1 - mar.alpha;
covcols = "cov_" + S.classes;
k0 = [0.15 0.15 0.15];

fig = figure('Units', 'inches', 'Position', [0.5 0.5 S.col 4.6], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% (a) per-class coverage
ax = nexttile(t); hold(ax, 'on');
hNom = plot(ax, [0.68 1], [0.68 1], ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8);
hc = gobjects(1, 6);
for k = 1:6
    c = S.okabe(k, :);
    plot(ax, nom, mar.(char(covcols(k))), '--', 'Color', c, 'LineWidth', S.lw);
    hc(k) = plot(ax, nom, lab.(char(covcols(k))), '-', 'Color', c, 'LineWidth', S.lw);
end
% legend-only handles (NaN data: nothing drawn)
hMar = plot(ax, nan, nan, '--', 'Color', k0, 'LineWidth', S.lw);
hLab = plot(ax, nan, nan, '-',  'Color', k0, 'LineWidth', S.lw);
hDef = plot(ax, nan, nan, 'o', 'LineStyle', 'none', 'Color', k0, 'MarkerSize', 3, 'MarkerFaceColor', k0);
hSiz = plot(ax, nan, nan, 's', 'LineStyle', 'none', 'Color', S.gray, 'MarkerSize', 3.2, 'MarkerFaceColor', 'w');
xlim(ax, [0.68 1]); ylim(ax, [0 1.02]);
xlabel(ax, 'Nominal coverage 1 - \alpha'); ylabel(ax, 'Class votes covered');
title(ax, '(a) Class-wise coverage');
hms_axes(ax, S);
axA = ax;

% (b) deferral and set size
ax = nexttile(t); hold(ax, 'on');
yyaxis(ax, 'left');
plot(ax, nom, mar.defer_rate, '--o', 'Color', k0, 'LineWidth', S.lw, 'MarkerSize', 3, 'MarkerFaceColor', k0);
plot(ax, nom, lab.defer_rate, '-o',  'Color', k0, 'LineWidth', S.lw, 'MarkerSize', 3, 'MarkerFaceColor', k0);
ylabel(ax, 'Fraction deferred'); ylim(ax, [0 1]);
yyaxis(ax, 'right');
plot(ax, nom, mar.set_size, '--s', 'Color', S.gray, 'LineWidth', S.lw, 'MarkerSize', 3.2, 'MarkerFaceColor', 'w');
plot(ax, nom, lab.set_size, '-s',  'Color', S.gray, 'LineWidth', S.lw, 'MarkerSize', 3.2, 'MarkerFaceColor', 'w');
ylabel(ax, 'Mean set size (classes)'); ylim(ax, [1 6]);
ax.YAxis(1).Color = 'k'; ax.YAxis(2).Color = 'k';
xlim(ax, [0.68 1]);
xlabel(ax, 'Nominal coverage 1 - \alpha');
title(ax, '(b) Deferral and set size');
hms_axes(ax, S);

% one legend under both panels, filled row by row (classes | rules | quantities of panel b)
h = [hc, hNom, hMar, hLab, hDef, hSiz];
lbl = [S.labels, "Nominal", "Marginal", "Label-cond.", "Deferral (b)", "Set size (b)"];
lg = hms_legend(axA, h, lbl, S, 3);
lg.Orientation = 'horizontal';
lg.NumColumns = 3;
hms_export(fig, outdir, 'conformal');
end
