%%Algoritmo t-SNE aplicado ao Dataset da UCI
clear; clc; close all;

T = readtable(['student/student-por.csv']); % Importacao do arquivo do dataset
A = table2cell(T);
i = 1;

% Autor do codigo para alterar os valores nominais Stephen
% (https://www.mathworks.com/matlabcentral/answers/383140-replace-string-value-in-cell-with-numerical-value)
idn = cellfun(@isnumeric,A); % identify numeric values.
out = nan(size(A));          % preallocate output matrix.
out(idn) = [A{idn}];         % allocate numeric values.
tmp = A(~idn);               % subset with char vectors.
vec = str2double(tmp);       % attempt to convert to numeric.
idx = isnan(vec);            % identify char not converted.
C = {'GP', 'MS', 'F', 'M', 'R', 'U', 'GT3', 'LE3', 'T', 'A', 'at_home', 'other', 'services', 'health', 'teacher', 'home', 'reputation', 'course', 'yes', 'no', 'father', 'mother'};
V = [1 , 0 , 1 , 0 , 0 , 1 , 1 , 0 , 1 , 0 , 0 , 1 , 2 , 3 , 4 , 1 , 2 , 3 , 1 , 0 , 1 , 0];
[idm,idc] = ismember(tmp(idx),C); % lookup table.
assert(all(idm),'Not in C:%s',sprintf(' %s,',C{~idm}))
vec(idx) = V(idc);
out(~idn) = vec;

% Calculo da aprovacao
atribute = 33;
table_size = length(table2array(T(:,atribute)));
while(i < table_size)
    if(T{i,atribute} <= 12)
        T.L(i) = "Reproved";
    else
        T.L(i) = "Approved";
    end
    i = i+1;
end

% Atributo a ser comparado
factor = table2array(T(:, "L"));
rng('default') % for reproducibility

% Metodo do t-SNE com a distancia de Minkowski e perplexidade 150
Y = tsne(out,'Algorithm','exact','Distance','minkowski','Perplexity',150);
gscatter(Y(:,1),Y(:,2),factor)
title('Approvation in Mathematics with Minkowski Distance')

% Calculo da perda do algoritmo
[Y,loss] = tsne(out,'Algorithm','exact','Distance','minkowski');
fprintf('2-D embedding has loss %g\n',loss)
