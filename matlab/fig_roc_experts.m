function fig_roc_experts(datadir, outdir)
%FIG_ROC_EXPERTS  Model versus individual experts, one-vs-rest against the leave-one-out expert consensus.
%   Row 1: ROC curves (FPR axis up to 0.5, which contains all expert operating points), AUROC printed in the
%          empty lower-right corner.
%   Row 2: precision-recall curves with the no-skill level (class prevalence), AUPRC printed in the empty
%          lower-left corner.
%   The average expert's operating point (x) is evaluated against the same reference. One legend, own tile.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S = hms_style();
R  = hms_read(datadir, 'roc_curves.csv');
A  = hms_read(datadir, 'roc_auc.csv');
E  = hms_read(datadir, 'expert_points.csv');
PR = hms_read(datadir, 'pr_curves.csv');
PA = hms_read(datadir, 'pr_auc.csv');
PE = hms_read(datadir, 'expert_pr_points.csv');
models = struct('method', {"kl", "pheads", "rate"}, 'backbone', {"spec2d", "spec2d", "cnn1d"}, ...
                'style', {'-', '-', '--'}, ...
                'label', {"Two-stage, spectrogram CNN", "Pool-heads, spectrogram CNN", "RaTE, 1-D CNN"});
xmax = 0.5;

fig = figure('Units', 'inches', 'Position', [0.5 0.5 S.wide 3.55], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig, 2, 6, 'TileSpacing', 'compact', 'Padding', 'compact');
% ---- row 1: ROC ------------------------------------------------------------------------------------------
for k = 1:6
    ax = nexttile(t, k); hold(ax, 'on');
    plot(ax, [0 xmax], [0 xmax], ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.5);
    h = gobjects(1, numel(models));
    for j = 1:numel(models)
        m = models(j); c = S.okabe(j, :);
        sel = R.method == m.method & R.backbone == m.backbone & R.cls == S.classes(k) & R.fpr <= xmax;
        h(j) = plot(ax, R.fpr(sel), R.tpr(sel), m.style, 'Color', c, 'LineWidth', S.lw);
        auc = A.auroc(A.method == m.method & A.backbone == m.backbone & A.cls == S.classes(k));
        text(ax, 0.98, 0.04 + 0.12 * (numel(models) - j), sprintf('%.3f', auc), 'Units', 'normalized', ...
             'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', 'Color', c, ...
             'FontName', S.font, 'FontSize', S.fsNote);
    end
    e = E(E.cls == S.classes(k), :);
    hE = plot(ax, e.fpr_expert, e.tpr_expert, 'kx', 'MarkerSize', 6, 'LineWidth', 1.2);
    axis(ax, [0 xmax 0 1]);
    xticks(ax, 0:0.25:xmax); yticks(ax, 0:0.25:1);
    title(ax, S.labels(k));
    xlabel(ax, 'FPR');
    if k == 1, ylabel(ax, 'TPR'); hLeg = [h, hE]; axLeg = ax; end
    hms_axes(ax, S);
end
% ---- row 2: precision-recall -------------------------------------------------------------------------------
for k = 1:6
    ax = nexttile(t, 6 + k); hold(ax, 'on');
    prev = PA.prevalence(PA.cls == S.classes(k)); prev = prev(1);
    hNS = plot(ax, [0 1], [prev prev], ':', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.7);
    for j = 1:numel(models)
        m = models(j); c = S.okabe(j, :);
        sel = PR.method == m.method & PR.backbone == m.backbone & PR.cls == S.classes(k);
        plot(ax, PR.recall(sel), PR.precision(sel), m.style, 'Color', c, 'LineWidth', S.lw);
        ap = PA.auprc(PA.method == m.method & PA.backbone == m.backbone & PA.cls == S.classes(k));
        text(ax, 0.03, 0.10 + 0.11 * (numel(models) - j), sprintf('%.3f', ap), 'Units', 'normalized', ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', 'Color', c, ...
             'FontName', S.font, 'FontSize', S.fsNote);
    end
    e = PE(PE.cls == S.classes(k), :);
    plot(ax, e.tpr_expert, e.ppv_expert, 'kx', 'MarkerSize', 6, 'LineWidth', 1.2);
    axis(ax, [0 1 0 1]);
    xticks(ax, 0:0.5:1); yticks(ax, 0:0.25:1);
    xlabel(ax, 'Recall');
    if k == 1, ylabel(ax, 'Precision'); hLeg = [hLeg, hNS]; end %#ok<AGROW>
    hms_axes(ax, S);
end
hms_legend(axLeg, hLeg, [[models.label], "Average expert", "No skill (prevalence)"], S, 5);
hms_export(fig, outdir, 'roc_experts');
end
