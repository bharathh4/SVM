
clc
clear all
close all



%Class 1
class_1_x1=[1+1*randn(1,1000) 3+1*randn(1,1000)];
class_1_x2=[3+1*randn(1,1000) 5+1*randn(1,1000)];
Class1_X=[class_1_x1 ;class_1_x2 ];
Class1_X=transpose(Class1_X);%Each feature in a column
c1=[Class1_X ones(length(Class1_X),1)];
%Class 2
class_2_x1=[5+1*randn(1,1000) 7+1*randn(1,1000)];
class_2_x2=[7+1*randn(1,1000) 9+1*randn(1,1000)];
Class2_X=[class_2_x1 ;class_2_x2 ];
Class2_X=transpose(Class2_X);%Each feature in a column
c2=[Class2_X 2*ones(length(Class1_X),1)];
%Class 3
class_3_x1=[6+1*randn(1,1000) 6.5+1*randn(1,1000)];
class_3_x2=[1+1*randn(1,1000) 3+1*randn(1,1000)];
Class3_X=[class_3_x1 ;class_3_x2 ];
Class3_X=transpose(Class3_X);%Each feature in a column
c3=[Class3_X 3*ones(length(Class1_X),1)];

%%
load SVMstructClass1Class2.mat

features=[c1;c2];
predicted=[];
actual=[];
for i=1:length(features)
predicted=[predicted svmclassify(SVMstruct12,features(i,1:2))];
actual=[actual features(i,3)];
end


confusionMatrix=confusionmat(predicted,actual);
normal=(1/2000)*confusionMatrix

%%


load SVMstructClass2Class3.mat
clear features
clear confusionMatrix
clear normal

features=[c2;c3];

predicted=[];
actual=[];
for i=1:length(features)
predicted=[predicted svmclassify(SVMstruct23,features(i,1:2))];
actual=[actual features(i,3)];
end


confusionMatrix=confusionmat(predicted,actual)
normal=(1/2000)*confusionMatrix



%%
load SVMstructClass1Class3.mat

clear features
clear confusionMatrix
clear normal


features=[c1;c3];

predicted=[];
actual=[];
for i=1:length(features)
predicted=[predicted svmclassify(SVMstruct13,features(i,1:2))];
actual=[actual features(i,3)];
end


confusionMatrix=confusionmat(predicted,actual)
normal=(1/2000)*confusionMatrix



%%
% normal =
% 
%     0.9585    0.0345
%     0.0415    0.9655
% 
% 
% confusionMatrix =
% 
%         1987          16
%           13        1984
% 
% 
% normal =
% 
%     0.9935    0.0080
%     0.0065    0.9920
% 
% 
% confusionMatrix =
% 
%         1973          29
%           27        1971
% 
% 
% normal =
% 
%     0.9865    0.0145
%     0.0135    0.9855

%%