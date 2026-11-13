function fig_noise_floor(datadir, outdir)
%FIG_NOISE_FLOOR  Noise floor versus number of votes (one column).
%   Empirical-Bayes floor (expected KL of a model that predicts the true panel distribution exactly),
%   the asymptotic approximation for reference, and the out-of-fold KL of the spectrogram CNN trained with
%   the two-stage curriculum and with pool-heads (mean over 3 seeds, patient-bootstrap 95% CI per bin).
%   The shaded band is the gap that remains reducible above the floor.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S = hms_style();
T = hms_read(datadir, 'noise_floor_by_n.csv');
bins = unique(T.n_lo);                         % 10..19 and 20 (= 20 or more votes)
lab = string(bins); lab(end) = "\geq" + string(bins(end));
x = 1:numel(bins);
val = @(meth, col) arrayfun(@(b) T.(col)(T.method == meth & T.n_lo == b), bins)';
ref = "pheads_spec2d";
floor_eb = val(ref, 'floor_eb'); floor_as = val(ref, 'floor_asym');
models = struct('method', {"kl_spec2d", "pheads_spec2d"}, 'color', {S.okabe(1, :), S.okabe(2, :)}, ...
                'marker', {'s', 'o'}, 'dx', {-0.12, 0.12}, ...
                'label', {"Two-stage, spectrogram CNN", "Pool-heads, spectrogram CNN"});

fig = figure('Units', 'inches', 'Position', [0.5 0.5 3.5 2.75], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig, 1, 1, 'Padding', 'compact');
ax = nexttile(t); hold(ax, 'on');
kl_ph = val(ref, 'kl');
hGap = fill(ax, [x, fliplr(x)], [floor_eb, fliplr(kl_ph)], S.light, 'EdgeColor', 'none');
hEB = plot(ax, x, floor_eb, '-o', 'Color', 'k', 'LineWidth', S.lw, 'MarkerSize', 3, 'MarkerFaceColor', 'k');
hAS = plot(ax, x, floor_as, ':', 'Color', [0.35 0.35 0.35], 'LineWidth', S.lw);
h = gobjects(1, numel(models));
for j = 1:numel(models)
    m = models(j);
    y = val(m.method, 'kl'); lo = val(m.method, 'kl_lo'); hi = val(m.method, 'kl_hi');
    h(j) = errorbar(ax, x + m.dx, y, y - lo, hi - y, ['-' m.marker], 'Color', m.color, 'LineWidth', 0.8, ...
                    'MarkerSize', 3.2, 'MarkerFaceColor', m.color, 'CapSize', 2);
end
xlim(ax, [0.5 numel(bins) + 0.5]); ylim(ax, [0 inf]);
xticks(ax, x); xticklabels(ax, lab); xtickangle(ax, 0);
xlabel(ax, 'Number of expert votes per event');
ylabel(ax, 'KL divergence (nats)');
hms_axes(ax, S);
hms_legend(ax, [h, hEB, hAS, hGap], [[models.label], "Noise floor (empirical Bayes)", ...
           "Asymptotic approximation", "Reducible gap (pool-heads)"], S, 2);
ev = arrayfun(@(b) T.events(T.method == ref & T.n_lo == b), bins);
fprintf('  events per bin: %s\n', mat2str(ev'));
hms_export(fig, outdir, 'noise_floor');
end
