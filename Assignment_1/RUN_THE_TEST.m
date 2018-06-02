clear all 
close all
clc

data = load('hw1data.mat');
A = data.Bdata;
y=test(A(:,1:10));
A(:,12)=y(:);
errorCounter_QC=0;
for i=1:length(A)
if A(i,11)~=A(i,12)
    errorCounter_QC=errorCounter_QC+1;
end
end
accuracy=(length(A)-errorCounter_QC)/length(A)