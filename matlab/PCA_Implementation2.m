%%Algoritmo PCA aplicado ao Dataset da Kaggle
clear; clc; close all;

T = readtable('StudentsPerformance/StudentsPerformance.csv');% Importacao do arquivo do dataset
A = table2cell(T);
i = 1;

% Autor do codigo Mathworks
% (https://www.mathworks.com/help/stats/pca.html)
% (https://www.mathworks.com/help/stats/biplot.html)
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

% Vetor com os atributos rótulos do dataset
vb = {'gender','race/ethnicity','parental level of education','lunch','test preparation course','math score','reading score','writing score'};

[coeff,score,latent] = pca(out); % Retorna os coeficientes do componentes principais em forma de matriz
Xcentered = score*coeff'; % Novos dados formados a partir dos dados do dataset subtraindo as médias das colunas correspondentes
biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',vb);% Sera greado o gráfico com as princpais compoentes

