function fig_pr_experts(datadir, outdir)
%FIG_PR_EXPERTS  Supplementary Fig. S3: one-vs-rest precision-recall curves against the leave-one-out expert
%   consensus, with the average expert's (recall, precision) and the no-skill level (class prevalence).
%   AUPRC values are printed in the empty lower-left corner of each panel; the legend has its own tile.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S = hms_style();
R = hms_read(datadir, 'pr_curves.csv');
A = hms_read(datadir, 'pr_auc.csv');
E = hms_read(datadir, 'expert_pr_points.csv');
models = struct('method', {"kl", "pheads", "rate"}, 'backbone', {"spec2d", "spec2d", "cnn1d"}, ...
                'style', {'-', '-', '--'}, ...
                'label', {"Two-stage, spectrogram CNN", "Pool-heads, spectrogram CNN", "RaTE, 1-D CNN"});

fig = figure('Units', 'inches', 'Position', [0.5 0.5 S.wide 1.95], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig, 1, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:6
    ax = nexttile(t); hold(ax, 'on');
    prev = A.prevalence(A.cls == S.classes(k)); prev = prev(1);
    hNS = plot(ax, [0 1], [prev prev], ':', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.7);
    h = gobjects(1, numel(models));
    for j = 1:numel(models)
        m = models(j); c = S.okabe(j, :);
        sel = R.method == m.method & R.backbone == m.backbone & R.cls == S.classes(k);
        h(j) = plot(ax, R.recall(sel), R.precision(sel), m.style, 'Color', c, 'LineWidth', S.lw);
        ap = A.auprc(A.method == m.method & A.backbone == m.backbone & A.cls == S.classes(k));
        text(ax, 0.03, 0.10 + 0.11 * (numel(models) - j), sprintf('%.3f', ap), 'Units', 'normalized', ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Color', c, ...
             'FontName', S.font, 'FontSize', S.fsNote);
    end
    e = E(E.cls == S.classes(k), :);
    hE = plot(ax, e.tpr_expert, e.ppv_expert, 'kx', 'MarkerSize', 6, 'LineWidth', 1.2);
    axis(ax, [0 1 0 1]);
    xticks(ax, 0:0.5:1); yticks(ax, 0:0.25:1);
    title(ax, S.labels(k));
    hms_axes(ax, S);
    if k == 1, h1 = [h, hE, hNS]; ax1 = ax; end
end
xlabel(t, 'Recall', 'FontName', S.font, 'FontSize', S.fsLabel);
ylabel(t, 'Precision', 'FontName', S.font, 'FontSize', S.fsLabel);
hms_legend(ax1, h1, [[models.label], "Average expert", "No skill (prevalence)"], S, 5);
hms_export(fig, outdir, 'pr_experts');
end
