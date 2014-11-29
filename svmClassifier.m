
%%
clear all
clc
close all

%Class 1
class_1_x1=[1+1*randn(1,1000) 3+1*randn(1,1000)];
class_1_x2=[3+1*randn(1,1000) 5+1*randn(1,1000)];
Class1_X=[class_1_x1 ;class_1_x2 ];
Class1_X=transpose(Class1_X);%Each feature in a column


%Class 2
class_2_x1=[5+1*randn(1,1000) 7+1*randn(1,1000)];
class_2_x2=[7+1*randn(1,1000) 9+1*randn(1,1000)];
Class2_X=[class_2_x1 ;class_2_x2 ];
Class2_X=transpose(Class2_X);%Each feature in a column


% %Class 3
% class_3_x1=[6+1*randn(1,1000) 6.5+1*randn(1,1000)];
% class_3_x2=[1+1*randn(1,1000) 3+1*randn(1,1000)];
% Class3_X=[class_3_x1 ;class_3_x2 ];
% Class3_X=transpose(Class3_X);%Each feature in a column


% X= [Class1_X ;Class2_X ;Class3_X];
X= [Class1_X ;Class2_X];

% Y=[ones(2000,1) ;2*ones(2000,1);3*ones(2000,1)];
Y=[ones(2000,1);2*ones(2000,1)];


inputs=(X);
targets=transpose(Y);

SVMstruct12 = svmtrain(inputs,targets,'Kernel_Function','rbf');

save('SVMstructClass1Class2.mat','SVMstruct12');
%%
clear all
clc
close all

% %Class 1
% class_1_x1=[1+1*randn(1,1000) 3+1*randn(1,1000)];
% class_1_x2=[3+1*randn(1,1000) 5+1*randn(1,1000)];
% Class1_X=[class_1_x1 ;class_1_x2 ];
% Class1_X=transpose(Class1_X);%Each feature in a column
% 

%Class 2
class_2_x1=[5+1*randn(1,1000) 7+1*randn(1,1000)];
class_2_x2=[7+1*randn(1,1000) 9+1*randn(1,1000)];
Class2_X=[class_2_x1 ;class_2_x2 ];
Class2_X=transpose(Class2_X);%Each feature in a column


%Class 3
class_3_x1=[6+1*randn(1,1000) 6.5+1*randn(1,1000)];
class_3_x2=[1+1*randn(1,1000) 3+1*randn(1,1000)];
Class3_X=[class_3_x1 ;class_3_x2 ];
Class3_X=transpose(Class3_X);%Each feature in a column


% X= [Class1_X ;Class2_X ;Class3_X];
X= [Class2_X ;Class3_X];

% Y=[ones(2000,1) ;2*ones(2000,1);3*ones(2000,1)];
Y=[2*ones(2000,1);3*ones(2000,1)];


inputs=(X);
targets=transpose(Y);

SVMstruct23 = svmtrain(inputs,targets,'Kernel_Function','rbf');

save('SVMstructClass2Class3.mat','SVMstruct23');
%%

%%
clear all
clc
close all

%Class 1
class_1_x1=[1+1*randn(1,1000) 3+1*randn(1,1000)];
class_1_x2=[3+1*randn(1,1000) 5+1*randn(1,1000)];
Class1_X=[class_1_x1 ;class_1_x2 ];
Class1_X=transpose(Class1_X);%Each feature in a column


% %Class 2
% class_2_x1=[5+1*randn(1,1000) 7+1*randn(1,1000)];
% class_2_x2=[7+1*randn(1,1000) 9+1*randn(1,1000)];
% Class2_X=[class_2_x1 ;class_2_x2 ];
% Class2_X=transpose(Class2_X);%Each feature in a column


%Class 3
class_3_x1=[6+1*randn(1,1000) 6.5+1*randn(1,1000)];
class_3_x2=[1+1*randn(1,1000) 3+1*randn(1,1000)];
Class3_X=[class_3_x1 ;class_3_x2 ];
Class3_X=transpose(Class3_X);%Each feature in a column


% X= [Class1_X ;Class2_X ;Class3_X];
X= [Class1_X ;Class3_X];

% Y=[ones(2000,1) ;2*ones(2000,1);3*ones(2000,1)];
Y=[ones(2000,1);3*ones(2000,1)];


inputs=(X);
targets=transpose(Y);

SVMstruct13 = svmtrain(inputs,targets,'Kernel_Function','rbf');

save('SVMstructClass1Class3.mat','SVMstruct13');
%%

