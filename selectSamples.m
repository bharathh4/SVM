function [train_samples test_samples]=selectSamples(y,percentage_training,percentage_testing)
%    selectSamples()     - returns training and testing samples
%    percentage_training - percentage of input files to be used for training 
%    percentage_texting  - percentage of input files to be used for testing

%    Examples [train_samples test_samples]=selectSamples(y,70,30)

N=length(y);
train_num=round((percentage_training/100)*N);

index=randperm(N); % scrambles index numbers randomly for y
train_samples_index=index(1:train_num); % selects given percentage of training samples
train_samples=y(train_samples_index,:);
test_samples_index=index(train_num+1:N); % selects given percentage of testing samples
test_samples=y(test_samples_index,:);

% Dead code
% train_samples_index = randi(N,1,train_num)
% train_samples=y(train_samples_index,:);
% test_samples_index=findNumbersNotInSet(1:N,train_samples_index);
% test_samples=y(test_samples_index);
% Dead code

end

% Dead code
function y=findNumbersNotInSet(set,members)
% find numbers not in a set
% eg set=[1 2 3 4 5 6 7]
%    members=[2 5]
%    y=[1 3 4 6 7]

ismem=ismember(set,members);
y=set(~ismem);

end
% Dead code

