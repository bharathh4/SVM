function predicted=svm_classifyHighLevel(percentage_training,feature)



load SubSetNormalizedFeaturesSet2.mat
y=SubSetNormalizedFeaturesSet2;
clear SubSetNormalizedFeaturesSet2;

% load NormalizedFeaturesSet2.mat
% y=NormalizedFeaturesSet2;
% clear NormalizedFeaturesSet2;

num_features=size(y,2)-1

[class1, class2 ,class3]=prepareData(y);


percentage_testing=30;
[train_samples_class1 test_samples_class1]=selectSamples(class1,percentage_training,percentage_testing);


percentage_testing=30;
[train_samples_class2 test_samples_class2]=selectSamples(class2,percentage_training,percentage_testing);


percentage_testing=30;
[train_samples_class3 test_samples_class3]=selectSamples(class3,percentage_training,percentage_testing);

% Note that feature 1 is in the columns,feature num_features is in the columns 




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
%%


p12=svmclassify(SVMstruct12,feature);

p13=svmclassify(SVMstruct13,feature);

p23=svmclassify(SVMstruct23,feature);

predicted = combineBinaryDecisons(p12,p23,p13)

end