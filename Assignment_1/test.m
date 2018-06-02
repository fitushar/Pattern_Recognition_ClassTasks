function y=test(A)
%% Importing the Data
Hwk1_Data= load('hw1data.mat');
Data=Hwk1_Data.Bdata;

%% Split the impotanted data into Positive and Negative class
Data_PositiveClass=Data((Data(:,11)==1),:); 
Data_NegativeClass=Data((Data(:,11)==-1),:);


%% Finding the size of the positive samples and dividing the size to training and sample
%% Split the class in 75% Train And 25% Testing set.

%Spliting the Positive class
SizeOf_Data_PositiveClass=length(Data_PositiveClass);%Getting the Length of the Positive class
Size_Train_PositiveClass= 0.75*SizeOf_Data_PositiveClass;  %Storing 25% Data of Positive Class data as Train Set
Size_Test_PositiveClass= 0.25*SizeOf_Data_PositiveClass;  %Storing 25% Data of Positive Class data as Test Set

%Spliting the Negative class
SizeOf_Data_NegativeClass=length(Data_NegativeClass); %Getting the Length of the Positive class
Size_Train_NegativeClass= 0.75*SizeOf_Data_NegativeClass;  %Storing 25% Data of Negative Class data as Train Set
Size_Test_NegativeClass= 0.25*SizeOf_Data_NegativeClass;   %Storing 25% Data of Negative Class data as Test Set


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
    Test_Data=A(:,1:10);    %Adding Testing Set

%% Finding Prior Probability 
P_of_PositiveClass= ((length(Train_PositiveClass))/(length(Train_PositiveClass)+ length(Train_NegativeClass))); %%Calculating the Probability of Positive Class
     P_of_NegativeClass= (length(Train_NegativeClass)/(length(Train_PositiveClass)+length(Train_NegativeClass))); %% Calculating the Probability of Negative Class

%%KNN Classifier
SizeOf_TestData= length(Test_Data);
SizeOf_TrainData= length(Train_Data);
k=5;
KnnC_Lables=zeros(SizeOf_TrainData,2);
   
%// Loop through each point and do logic as seen above:
for ii = 1 : SizeOf_TestData

    %// Use Euclidean
    d = sqrt(sum(abs(Train_Data(:,1:10)-repmat(Test_Data(ii,1:10), [length(Train_Data) 1])).^2, 2)); %%Finding euclidean distance between
    
    %// Putting the Distance and True label in KnnC_Lables
    KnnC_Lables=[d,Train_Data(:,11)];
    
   
    %// Finding the K Nearest Classes
    KnnC_Lables_Sorted=sortrows(KnnC_Lables,1);
    knnC=KnnC_Lables_Sorted(1:k,2);
    
    %// Now Checking How many of KnnC belong to Positive and Negative Class
    Num_KnnC_Positive_Label=length(find(knnC==1));
    Num_KnnC_Negative_Label=length(find(knnC==-1));
    
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

y=Out_label';
  
