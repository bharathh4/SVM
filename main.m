%% Quick Test : Classify a single feature

clc
clear all
close all

load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

% load NormalizedFeaturesSet2.mat
% y=NormalizedFeaturesSet2;
% clear NormalizedFeaturesSet2;

num_features=size(y,2)-1

[class1, class2 ,class3]=prepareData(y);

percentage_training=30;
feature=class1(200,1:num_features);

label=svm_classifyHighLevel(percentage_training,feature)



%%

%% %% Calculate average confusion matrix.For a quick test set numIterations=2 and percentage_training=40


clc
clear all

num_Iterations=1;
percentage_training=70;
avgConfusion=statisticalAvgConfusionMatrix(num_Iterations,percentage_training)

% percentage_training=70;
% 
% avgConfusion =
% 
%     0.9960    0.0040         0
%          0    0.9996    0.0004
%     0.0058    0.0173    0.9769
%%



%% To visualize flow
clc
clear all
close all

load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

% load NormalizedFeaturesSet2.mat
% y=NormalizedFeaturesSet2;
% clear NormalizedFeaturesSet2;

num_features=size(y,2)-1

[class1, class2 ,class3]=prepareData(y);

percentage_training=70;
percentage_testing=30;
[train_samples_class1 test_samples_class1]=selectSamples(class1,percentage_training,percentage_testing);

percentage_training=70;
percentage_testing=30;
[train_samples_class2 test_samples_class2]=selectSamples(class2,percentage_training,percentage_testing);

percentage_training=70;
percentage_testing=30;
[train_samples_class3 test_samples_class3]=selectSamples(class3,percentage_training,percentage_testing);

% Note that feature 1 is in the columns,feature num_features is in the columns 

c1=[test_samples_class1 ones(length(test_samples_class1),1)];
c2=[test_samples_class2 2*ones(length(test_samples_class2),1)];
c3=[test_samples_class3 3*ones(length(test_samples_class3),1)];

testFeatures=[c1;c2;c3];

predicted=[];
actual=[];

%%
X= [train_samples_class1 ;train_samples_class2];

% Y=[ones(2000,1) ;2*ones(2000,1);3*ones(2000,1)];
Y=[ones(length(train_samples_class1),1);2*ones(length(train_samples_class2),1)];


inputs=(X);
targets=transpose(Y);

SVMstruct12 = svmtrain(inputs,targets,'Kernel_Function','rbf');
%%
X= [train_samples_class1 ;train_samples_class3];

% Y=[ones(2000,1) ;2*ones(2000,1);3*ones(2000,1)];
Y=[ones(length(train_samples_class1),1);3*ones(length(train_samples_class3),1)];


inputs=(X);
targets=transpose(Y);

SVMstruct13 = svmtrain(inputs,targets,'Kernel_Function','rbf');
%%
X= [train_samples_class2 ;train_samples_class3];

% Y=[ones(2000,1) ;2*ones(2000,1);3*ones(2000,1)];
Y=[2*ones(length(train_samples_class2),1);3*ones(length(train_samples_class3),1)];


inputs=(X);
targets=transpose(Y);

SVMstruct23 = svmtrain(inputs,targets,'Kernel_Function','rbf');
%% Testing
%%



features=[c1;c2];
predicted=[];
actual=[];
for i=1:length(features)
predicted=[predicted svmclassify(SVMstruct12,features(i,1:num_features))];
actual=[actual features(i,num_features+1)];
end


confusionMatrix=confusionmat(predicted,actual);
normal=(1/length(c1))*confusionMatrix;
normMat=transpose(normal)
%%



clear features
clear confusionMatrix
clear normal

features=[c2;c3];

predicted=[];
actual=[];
for i=1:length(features)
predicted=[predicted svmclassify(SVMstruct23,features(i,1:num_features))];
actual=[actual features(i,num_features+1)];
end


confusionMatrix=confusionmat(predicted,actual);
normal=(1/length(c1))*confusionMatrix;
normMat=transpose(normal)


%%

clear features
clear confusionMatrix
clear normal


features=[c1;c3];

predicted=[];
actual=[];
for i=1:length(features)
predicted=[predicted svmclassify(SVMstruct13,features(i,1:num_features))];
actual=[actual features(i,num_features+1)];
end


confusionMatrix=confusionmat(predicted,actual);
normal=(1/length(c1))*confusionMatrix;
normMat=transpose(normal)


%%
%  
% 1 and 2
% 
% normal =
% 
%     0.9996    0.0013
%     0.0004    0.9987
% 
% 1 and 3
% 
% confusionMatrix =
% 
%         2295          17
%            3        2281
% 
% 
% normal =
% 
%     0.9987    0.0074
%     0.0013    0.9926
% 
% 2 and 3
% 
% confusionMatrix =
% 
%         2219         174
%           79        2124
% 
% 
% normal =
% 
%     0.9656    0.0757
%     0.0344    0.9243

%      0.9996     0.0013           0.0074
%      0.0004     0.9987,0.9656    0.0757      
%      0.0013     0.0344           0.9243

