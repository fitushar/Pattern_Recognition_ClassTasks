%% Problem 1.4. Linear Classifier

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
numberOfitaration=10; % Number of iteration
Error_LC=0;           % Defining Initial Error=0
TPR_LC=0;             % Defining Initial True Positive Rate(TPR)=0
AUC_LC=0;             % Defining Initial Area Under Curve(AUC)=0
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
Size_Train_NegativeClass= 0.75*SizeOf_Data_NegativeClass;  %Storing 25% Data of Negative Class data as Train Set
Size_Test_NegativeClass= 0.25*SizeOf_Data_NegativeClass;   %Storing 25% Data of Negative Class data as Test Set

%% Iteration of the Classifier

for iteration=1:numberOfitaration
    
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
    
    %% ----Paramets for classifier----
      
    
    %% Calculating Mean Positive and Negative Class
    mean_PositiveClass= mean( Train_PositiveClass(:,1:10));
    mean_NegativeClass= mean( Train_NegativeClass(:,1:10));
    
    %% Calculating the Sigma
    sigma_PositiveClass=cov(Train_PositiveClass(:,1:10));
    sigma_NegatieClass=cov(Train_NegativeClass(:,1:10));
    
    %% Computing the Prior Probabilities
    P_of_PositiveClass= ((length(Train_PositiveClass))/(length(Train_PositiveClass)+ length(Train_NegativeClass))); %%Calculating the Probability of Positive Class
    P_of_NegativeClass= (length(Train_NegativeClass)/(length(Train_PositiveClass)+length(Train_NegativeClass))); %% Calculating the Probability of Negative Class
    
    %% Calculating Sigma for Linear Classifier
     sigma_of_LinearClassifer = ( P_of_PositiveClass*sigma_PositiveClass) + (P_of_NegativeClass*sigma_NegatieClass);

    %% Implementing Linear Classifer
    
    
    %% Calculating Discriminant Fuction.
    ErrorCounter_LC=0;
    for i=1:length(Test_Data)
        Feature_vector=Test_Data(i,1:10); %Taking the ith row of the Test set with 10 features.
        g1(i,:)=discriminantFunction(Feature_vector,mean_PositiveClass,sigma_of_LinearClassifer,P_of_PositiveClass); %%finding descriminat Function for the positive class for the ith sample
        g2(i,:)=discriminantFunction(Feature_vector,mean_NegativeClass,sigma_of_LinearClassifer,P_of_NegativeClass); %%finding descriminat Function for the positive class for the ith sample
        
      %% Making the decision
      Decision= g1(i,:)-g2(i,:);
      
  %Evaluating desion and Storing the Decision Data in Test set at 12th colum
      
      if Decision >0
         Test_Data(:,12)=1; %% Storing Desion as Positive Class=1 if Descion > 0
      else
         Test_Data(:,12)=-1; %% Storing Desion as Negative Class=-1 if Descion < 0
      end
      
     %% Now Making Matrix For ROC
     
     ROC_Matrix(i,1)=Test_Data(i,11);
     ROC_Matrix(i,2)=Decision;
     
     %% Incrementng ErrorCounter
     if Test_Data(i,11)~= Test_Data(i,12)
          ErrorCounter_LC=ErrorCounter_LC+1 ;  
     end
      
    end
  
  %% Ploting ROC Curve
  % Using [R,a]=EvalRoc(A,poslab,neglab)
     %A:array Nx2 with the results on N samples
        %Column 1 contains the True Label  
        %Column 2 contains the output of the classifier
     %Returns R=
     
  %R: array with the coordinates (FPR,TPR) of the points of the ROC curve on
      %column 1 and column 2; column 3 contains the threshold on the value of
      %the classifier that produces (FPR,TPR)
  %a: the area under the ROC curve (AUC)
  
  [R_LC,AUC_LC]=EvalROC(ROC_Matrix,+1,-1); %Getting the coordinates for ROC Plot
   Q=[];
   RChPlot(R_LC,Q,'ROC: Linear Classifier'); %Ploting the ROC
  
%% At a FPR of 0.1, TPR is found from the ROC
FPR_index = find(R_LC(:,1)==0.1); %% Taking all the rows having FPR=0.1
TPR_LC = sum(R_LC(FPR_index,2))/length(FPR_index); %True Postive Rate for FPR=0.1
    

%% Stroing All the Results of each Iteration in a Matrix
Performance_Parameters_Matrix(1,iteration)=(length(Test_Data)-ErrorCounter_LC)/length(Test_Data); %% Accuracy
Performance_Parameters_Matrix(2,iteration)=TPR_LC; %%Storing TPR
Performance_Parameters_Matrix(3,iteration)= AUC_LC; %%Stoing AUC

    
end
 
 %% Calculating the  Average and Storing in the (iteration+1) column.
 Performance_Parameters_Matrix(1,iteration+1)=mean(Performance_Parameters_Matrix(1,1:iteration));
 Performance_Parameters_Matrix(2,iteration+1)=mean(Performance_Parameters_Matrix(2,1:iteration));
 Performance_Parameters_Matrix(3,iteration+1)=mean(Performance_Parameters_Matrix(3,1:iteration));
 
 %%displaying all the performance parameters
 Performance_Parameters_Matrix
