function fig_confusion(datadir, outdir)
%FIG_CONFUSION  individual expert votes and the spectrogram pool-head model against the
%   leave-one-out consensus of the other experts (>=10-vote development events, seed 42).
%   Panels (a,b) normalised by consensus class (diagonal = recall); (c,d) by decision (diagonal = precision).
%   The colour bar sits in its own tile to the right; cell values are printed inside the cells.
%
%   Authors: Tran Anh Tu, Huynh Hoang Nhut
S = hms_style();
C = hms_read(datadir, 'confusion.csv');
whos_ = ["expert", "model"];
norms = ["row", "col"];
colTitle = ["Individual expert", "Pool-heads"];
rowTitle = ["recall", "precision"];
cmap = [linspace(1, 0, 256)' linspace(1, 0.447, 256)' linspace(1, 0.698, 256)'];  % white -> blue

fig = figure('Units', 'inches', 'Position', [0.5 0.5 S.wide 2.35], 'Color', 'w', 'Renderer', 'painters');
t = tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
panel = 'abcd'; q = 0;
for r = 1:2
    for w = 1:2
        ax = nexttile(t); q = q + 1;
        M = zeros(6);
        for i = 1:6
            for j = 1:6
                M(i, j) = C.pct(C.who == whos_(w) & C.norm == norms(r) & C.consensus == S.classes(i) & ...
                                C.decision == S.classes(j));
            end
        end
        imagesc(ax, M, [0 100]); colormap(ax, cmap);
        axis(ax, 'square');
        for i = 1:6
            for j = 1:6
                if M(i, j) > 60, tc = 'w'; else, tc = 'k'; end
                text(ax, j, i, sprintf('%.0f', M(i, j)), 'HorizontalAlignment', 'center', ...
                     'Color', tc, 'FontName', S.font, 'FontSize', 5.5);
            end
        end
        xticks(ax, 1:6); yticks(ax, 1:6);
        xticklabels(ax, S.labels); yticklabels(ax, S.labels); xtickangle(ax, 45);
        title(ax, sprintf('(%s) %s, %s', panel(q), colTitle(w), rowTitle(r)));
        if q == 1, ylabel(ax, 'Leave-one-out consensus'); else, yticklabels(ax, []); end
        xlabel(ax, 'Decision');
        hms_axes(ax, S);
        set(ax, 'TickLength', [0 0], 'Box', 'on');
    end
end
cb = colorbar(ax);
cb.Layout.Tile = 'east';
cb.Label.String = 'Percent of row (a,b) or column (c,d)';
cb.FontName = S.font; cb.FontSize = S.fs;
hms_export(fig, outdir, 'confusion');
end
