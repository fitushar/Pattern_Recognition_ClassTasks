
%% Problem 1.4. KNN Classifier

%% Defining All the variable used in this Code
 
 %numberOfitaration = Number of Iteratio   
 %Error_LC=Error of Linear Classifier
 %TPR_LC=True Positive Rate(TPR)=0
 %AUC_LC=Area Under Curve Classifier
 
 % Data_PositiveClass= All the Data of Positive Class.
 % Data_NegativeClass= All the Data of Negative Class.

%%
clc
clear all;
close all;
%% Defining User input Parameters
%numberOfitaration=2; % Number of iteration
Error_KNN=0;           % Defining Initial Error=0
TPR_KNN=0;             % Defining Initial True Positive Rate(TPR)=0
AUC_KNN=0;             % Defining Initial Area Under Curve(AUC)=0
%% Importing the Data
Hwk1_Data= load('hw1data.mat');
Data=Hwk1_Data.Bdata;

%% Split the impotanted data into Positive and Negative class
Data_PositiveClass=Data((Data(:,11)==1),:); 
Data_NegativeClass=Data((Data(:,11)==-1),:);

%% Split the class in 75% Train And 25% Testing set.

%Spliting the Positive class
SizeOf_Data_PositiveClass=length(Data_PositiveClass);%Getting the Length of the Positive class
Size_Train_PositiveClass= 0.75*SizeOf_Data_PositiveClass;  %Storing 25% Data of Positive Class data as Train Set
Size_Test_PositiveClass= 0.25*SizeOf_Data_PositiveClass;  %Storing 25% Data of Positive Class data as Test Set

%Spliting the Negative class
SizeOf_Data_NegativeClass=length(Data_NegativeClass); %Getting the Length of the Positive class
Size_Train_NegativeClass= 0.65*SizeOf_Data_NegativeClass;  %Storing 25% Data of Negative Class data as Train Set
Size_Test_NegativeClass= 0.35*SizeOf_Data_NegativeClass;   %Storing 25% Data of Negative Class data as Test Set

%% Iteration of the Classifier


    
    %% Now Randomly Select 75% Traing and 25% Testing Set of Positive class
    IndexOf_Test_PositiveClass=randsample(1:SizeOf_Data_PositiveClass,Size_Test_PositiveClass ); %Creating Logical Index value of Test Set of Positive class
    Test_PositiveClass=Data_PositiveClass(IndexOf_Test_PositiveClass,:); %%Saving Postive Class Test Set
    
    IndexOf_Train_PositiveClass= setdiff(1:SizeOf_Data_PositiveClass,IndexOf_Test_PositiveClass); %%Creating Logical Index value of Train Set of Positive class
    Train_PositiveClass=Data_PositiveClass(IndexOf_Train_PositiveClass,:); %Saving the Positive Train Set
    
    %% Now Randomly Select 75% Traing and 25% Testing Set of Negative Class
    IndexOf_Test_NegativeClass=randsample(1:SizeOf_Data_NegativeClass,Size_Test_NegativeClass);%Creating Logical Index value of Test Set of Negative class
    Test_NegativeClass =Data_NegativeClass(IndexOf_Test_NegativeClass,:);%%Saving Negative Class Test Set
    
    IndexOf_Train_NegativeClass=setdiff(1:SizeOf_Data_NegativeClass,IndexOf_Test_NegativeClass); %%Creating Logical Index value of Train Set of Negative class
    Train_NegativeClass =Data_NegativeClass(IndexOf_Train_NegativeClass,:); %%%Saving Negative Class Test Set
    
    %% Combining Postive Class and Negative Class Train and Test sets to make final Trianing and Testing Test
    
    Train_Data=[Train_PositiveClass;Train_NegativeClass]; %Adding Traing Set
    Test_Data=[Test_PositiveClass;Test_NegativeClass];    %Adding Testing Set

%%
%     %% ----Paramets for classifier----
%       
%     
%     %% Calculating Mean Positive and Negative Class
%     mean_PositiveClass= mean( Train_PositiveClass(:,1:10));
%     mean_NegativeClass= mean( Train_NegativeClass(:,1:10));
%     
%     %% Calculating the Sigma
%     sigma_PositiveClass=cov(Train_PositiveClass(:,1:10));
%     sigma_NegatieClass=cov(Train_NegativeClass(:,1:10));
%     
     %% Computing the Prior Probabilities
     P_of_PositiveClass= ((length(Train_PositiveClass))/(length(Train_PositiveClass)+ length(Train_NegativeClass))); %%Calculating the Probability of Positive Class
     P_of_NegativeClass= (length(Train_NegativeClass)/(length(Train_PositiveClass)+length(Train_NegativeClass))); %% Calculating the Probability of Negative Class
%     
%     %% Calculating Sigma for Linear Classifier
%      sigma_of_LinearClassifer = ( P_of_PositiveClass*sigma_PositiveClass) + (P_of_NegativeClass*sigma_NegatieClass);

    %% Implementing Linear Classifer

SizeOf_TestData= length(Test_Data);
SizeOf_TrainData= length(Train_Data);
k = 1;
k_max=100;
KnnC_Lables=zeros(SizeOf_TrainData,2);
for knn_iteration=1:k_max
    k=knn_iteration;
   
%Loop through each point and do logic as seen above:
for ii = 1 : SizeOf_TestData

    %Use Euclidean
    d = sqrt(sum(abs(Train_Data(:,1:10)-repmat(Test_Data(ii,1:10), [length(Train_Data) 1])).^2, 2)); %%Finding euclidean distance between
    
    %Putting the Distance and True label in KnnC_Lables
    KnnC_Lables=[d,Train_Data(:,11)];
    
   
    %Finding the K Nearest Classes
    KnnC_Lables_Sorted=sortrows(KnnC_Lables,1);
    knnC=KnnC_Lables_Sorted(1:k,2);
    
    %Now Checking How many of KnnC belong to Positive and Negative Class
    Num_KnnC_Positive_Label=length(find(knnC==1));
    Num_KnnC_Negative_Label=length(find(knnC==-1));
    
    %This ROC matrix is for evaluating ROC later on if needed
    ROC_Matrix(ii,1)=Test_Data(ii,11);
    ROC_Matrix(ii,2)=Num_KnnC_Positive_Label/k;
    
    %%Getting the class level
    if ( Num_KnnC_Positive_Label > Num_KnnC_Negative_Label)
        
           Lables=1;
    elseif( Num_KnnC_Positive_Label == Num_KnnC_Negative_Label)
       if (P_of_PositiveClass>P_of_NegativeClass)
           Lables=1;
       else
           Lables=-1;
       end
    else
      Lables=-1;
    end
    
    Out_label(ii)=Lables;    
end

Out_label=Out_label';

%ploting the ROC
[R_KNN,AUC_KNN]=EvalROC(ROC_Matrix,1,-1);

Q=[];
RChPlot(R_KNN,Q,'k-nn Classifier')

% Error counter
ErrorKnn=0;
for i=1:length(Test_Data)
    if(Out_label(i)~=Test_Data(i,11))
        ErrorKnn=ErrorKnn+1;
    end
end
%%
upper_point = find(R_KNN(:,1)<0.1, 1, 'last'); %%Finding the closest point 
if(R_KNN(upper_point,1)<0.1) %%Checking whether the next closet point is an index higher or lower
below_point = upper_point + 1; %%if higher
else
    below_point=upper_point-1; %%if lower
end
%computing gradient fpr getting the perfect FPR
Gradient=(R_KNN(below_point,2)-R_KNN(upper_point,2))/(R_KNN(below_point,1)-R_KNN(upper_point,1)); %%gradient=y2-y1/x2-x1
TPR_KNN=Gradient*(0.1)-(Gradient*R_KNN(upper_point,1))+(R_KNN(upper_point,2)); %%y=mx-mx1+y1, where y=TRP with x=0.1 FPR



%%table to store all performance parameters at different values of k
table_knn(1,knn_iteration)=(length(Test_Data)-ErrorKnn)/length(Test_Data);

table_knn(2,knn_iteration)=TPR_KNN;

table_knn(3,knn_iteration)=AUC_KNN;

end

%%storing average at the last column of each performance parameter
table_knn(1,knn_iteration+1)=mean(table_knn(1,1:knn_iteration));

table_knn(2,knn_iteration+1)=mean(table_knn(2,1:knn_iteration));

table_knn(3,knn_iteration+1)=mean(table_knn(3,1:knn_iteration));



    
 
    
