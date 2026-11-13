function fig_eeg_examples(datadir, outdir)
%FIG_EEG_EXAMPLES  three development events selected by fixed rules
%   (see scripts/14_export_figdata.py): (a) confident, (b) deferred, (c) 3-7-vote pool shift.
%   Top: central 20 s of the 18-channel bipolar EEG (the 10-s labelled epoch shaded).
%   Bottom: expert vote shares, the >=10-vote head pi_0 and, for (c), the 3-7-vote head pi_1;
%   triangles mark the classes in the label-conditional prediction set (alpha = 0.2).
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S = hms_style();
M = hms_read(datadir, 'eeg_examples_meta.csv');
chanNames = ["Fp1-F7","F7-T3","T3-T5","T5-O1","Fp2-F8","F8-T4","T4-T6","T6-O2","Fp1-F3","F3-C3", ...
             "C3-P3","P3-O1","Fp2-F4","F4-C4","C4-P4","P4-O2","Fz-Cz","Cz-Pz"];
cases = ["A", "B", "C"];
titles = ["(a) Confident", "(b) Deferred", "(c) 3-7-vote pool"];
showSec = 10;                                   % half-width of the displayed window (s)

fig = figure('Units', 'inches', 'Position', [0.5 0.5 S.wide 4.6], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig, 5, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
bLeg = []; hmLeg = []; axLeg = [];
for j = 1:3
    r = M(M.example == cases(j), :);
    if isempty(r), warning('case %s missing in eeg_examples_meta.csv', cases(j)); continue; end
    E = readmatrix(fullfile(datadir, "eeg_case_" + cases(j) + ".csv"));
    tt = E(:, 1); X = E(:, 2:end);
    keep = abs(tt) <= showSec; tt = tt(keep); X = X(keep, :);
    sp = 50 * ceil(4 * median(std(X, 0, 1)) / 50);   % channel spacing, rounded to 50 uV
    sp = max(sp, 50);

    % ---- EEG traces --------------------------------------------------------------------------------
    ax = nexttile(t, j, [3 1]); hold(ax, 'on');
    yTop = 0.5 * sp; yBot = -(17.5) * sp;
    patch(ax, [-5 5 5 -5], [yBot yBot yTop yTop], [1 0.96 0.80], 'EdgeColor', 'none');
    for ch = 1:18
        plot(ax, tt, X(:, ch) - (ch - 1) * sp, '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 0.35);
    end
    xlim(ax, [-showSec showSec]); ylim(ax, [yBot yTop]);
    yticks(ax, fliplr(-(0:17) * sp));
    if j == 1
        yticklabels(ax, fliplr(chanNames));
    else
        yticklabels(ax, []);
    end
    xticks(ax, -showSec:5:showSec);
    xlabel(ax, 'Time (s)');
    title(ax, {titles(j), sprintf('pool %d, %d votes; spacing %d \\muV', r.pool, r.n, sp)});
    hms_axes(ax, S);
    ax.YAxis.FontSize = 5.5;

    % ---- votes and predicted distributions -------------------------------------------------------------
    ax = nexttile(t, 9 + j, [2 1]); hold(ax, 'on');
    v  = arrayfun(@(c) r.(char("votes_" + c)), S.classes) / r.n;
    p0 = arrayfun(@(c) r.(char("p0_" + c)), S.classes);
    p1 = arrayfun(@(c) r.(char("p1_" + c)), S.classes);
    inset = arrayfun(@(c) r.(char("inset_" + c)), S.classes) > 0;
    Y = [v; p0];
    cols = [S.light * 0.85; S.okabe(1, :)];
    if all(isfinite(p1)), Y = [Y; p1]; cols = [cols; S.okabe(2, :)]; end %#ok<AGROW>
    b = bar(ax, 1:6, Y', 0.85, 'grouped', 'EdgeColor', 'none');
    for q = 1:numel(b), b(q).FaceColor = cols(q, :); end
    yMark = 1.06;
    hm = plot(ax, find(inset), yMark * ones(1, nnz(inset)), 'kv', 'MarkerSize', 4, 'MarkerFaceColor', 'k');
    xlim(ax, [0.4 6.6]); ylim(ax, [0 1.15]);
    xticks(ax, 1:6); xticklabels(ax, S.labels); xtickangle(ax, 0);
    if j == 1, ylabel(ax, 'Probability / vote share'); end
    status = "reported";
    if r.deferred, status = "deferred"; end
    title(ax, "Set: \{" + strjoin(S.labels(inset), ", ") + "\}, " + status);
    hms_axes(ax, S);
    ax.XAxis.FontSize = 5.5;
    if numel(b) > numel(bLeg), bLeg = b; hmLeg = hm; axLeg = ax; end   % legend handles from one axes
end
if ~isempty(bLeg)
    names = ["Expert vote share", "\pi_0 (\geq10-vote head)", "\pi_1 (3-7-vote head)"];
    hms_legend(axLeg, [bLeg, hmLeg], [names(1:numel(bLeg)), "Class in prediction set"], S, 4);
end
hms_export(fig, outdir, 'eeg_examples');
end
