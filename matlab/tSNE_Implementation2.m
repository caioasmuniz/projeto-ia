%%Algoritmo t-SNE aplicado ao Dataset Kaggle
clear; clc; close all;

T = readtable('StudentsPerformance/StudentsPerformance.csv'); % Importacao do arquivo do dataset
A = table2cell(T);

% Autor do codigo para alterar os valores nominais Stephen
% (https://www.mathworks.com/matlabcentral/answers/383140-replace-string-value-in-cell-with-numerical-value)
idn = cellfun(@isnumeric,A); % identify numeric values.
out = nan(size(A));          % preallocate output matrix.
out(idn) = [A{idn}];         % allocate numeric values.
tmp = A(~idn);               % subset with char vectors.
vec = str2double(tmp);       % attempt to convert to numeric.
idx = isnan(vec);            % identify char not converted.
C = {'female', 'male', 'group A', 'group B', 'group C', 'group D', 'group E', 'standard', 'free/reduced', 'none', 'completed', 'some high school', 'high school', 'some college', char("associate's degree"), char("bachelor's degree"), char("master's degree")};
V = [1, 0, 1, 2, 3, 4, 5, 0, 1, 0, 1, 0, 1, 2, 3, 4, 5];
[idm,idc] = ismember(tmp(idx),C); % lookup table.
assert(all(idm),'Not in C:%s',sprintf(' %s,',C{~idm}))
vec(idx) = V(idc);
out(~idn) = vec;

% Calculo da media e classificacao dos alunos
i = 1;
atribute = 6; % Atributo a ser considerado (6 = Math, 7 = Reading, 8 = Writing)
atribute_mean = mean(table2array(T(:,atribute)));
table_size = length(table2array(T(:,atribute)));
while(i < table_size)
    if(T{i,atribute} <= atribute_mean)
        T.L(i) = "Below Average";
    else
        T.L(i) = "Above Average";
    end
    i = i+1;
end

% Atributo a ser comparado
factor = table2array(T(:, "L"));
rng('default') % for reproducibility

% Metodo do t-SNE com a distancia de Minkowski e perplexidade 50
Y = tsne(out,'Algorithm','exact','Distance','minkowski','Perplexity', 50);
subplot(3,1,1)
gscatter(Y(:,1),Y(:,2),factor)
title('Writing score with Minkowski Distance')
subtitle('Perplexity = 50')

% Metodo do t-SNE com a distancia de Minkowski e perplexidade 100
Y = tsne(out,'Algorithm','exact','Distance','minkowski','Perplexity', 100);
subplot(3,1,2)
gscatter(Y(:,1),Y(:,2),factor)
subtitle('Perplexity = 100')

% Metodo do t-SNE com a distancia de Minkowski e perplexidade 150
Y = tsne(out,'Algorithm','exact','Distance','minkowski','Perplexity', 150);
subplot(3,1,3)
gscatter(Y(:,1),Y(:,2),factor)
subtitle('Perplexity = 150')

% Calculo da perda do algoritmo
[Y,loss] = tsne(out,'Algorithm','exact','Distance','minkowski');
fprintf('2-D embedding has loss %g\n',loss)