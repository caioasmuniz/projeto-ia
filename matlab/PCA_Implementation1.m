%%Algoritmo PCA aplicado ao Dataset da UCI
clear; clc; close all;

T = readtable('student/student-por.csv');% Importacao do arquivo do dataset
A = table2cell(T);

% Autor do codigo Mathworks
% (https://www.mathworks.com/help/stats/pca.html)
% (https://www.mathworks.com/help/stats/biplot.html)
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

% Vetor com os atributos rótulos do dataset
vb = {'school', 'sex', 'age', 'adress','famsize', 'Pstatus', 'Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3'};

[coeff,score,latent] = pca(out); % Retorna os coeficientes do componentes principais em forma de matriz
Xcentered = score*coeff'; % Novos dados formados a partir dos dados do dataset subtraindo as médias das colunas correspondentes
biplot(coeff(:,1:2),'scores',score(:,1:2),'varlabels',vb);% Sera greado o gráfico com as princpais compoentes

