function [R,a]=EvalRoc(A,poslab,neglab)
% [R,a]=EvalRoc(A,poslab,neglab)
%
% Calculates the points of the ROC curve
% *** Arguments ***
% A: array Nx2 with the results on N samples
%   Column 1 contains the label of the sample (poslab for positive class, neglab for negative class)  
%   Column 2 contains the output of the classifier
% poslab: the label for positive class (poslab is assumed +1 if no value is given)
% neglab: the label for negative class (neglab is assumed -1 if no value is given)
% *** Output ***
% R: array with the coordinates (FPR,TPR) of the points of the ROC curve on
%   column 1 and column 2; column 3 contains the threshold on the value of
%   the classifier that produces (FPR,TPR)
% a: the area under the ROC curve (AUC)

if nargin < 3
    neglab=-1;
end
if nargin == 1
    poslab=1;    
end

% Calcolo il numero di campioni positivi e negativi
p=length(find(A(:,1)==poslab));
n=length(find(A(:,1)==neglab));

% Inizializzo la matrice R (che accoglie i punti della curva ROC) ed i vari contatori
R=[];
fcount=0;
tcount=0;
outlast=-Inf;

% Riordino l'array secondo i valori in col. 2 in ordine decrescente
E=flipud(sortrows(A,2));
%a2=0;
for i=1:length(E(:,1))
    if(E(i,2)~=outlast)
        outlast=E(i,2);
        R=[R; fcount/n tcount/p outlast]; % Metto anche il valore della threshold
    end
    
    if(E(i,1)==poslab)
        tcount=tcount+1;
    elseif(E(i,1)==neglab)
        fcount=fcount+1;
     %   a2=a2+tcount/(n*p);
    end
end

if(outlast > 0)
    outlast=0;
else
    outlast=outlast-1e-4;
end

R=[R; fcount/n tcount/p outlast];

a=0;
for i=2:length(R)
    a=a+(R(i,2)+R(i-1,2))*(R(i,1)-R(i-1,1))/2;
end
