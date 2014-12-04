function [class1, class2 ,class3]=prepareData(y)

classLabelIndex=size(y,2);
num_features=size(y,2)-1;

c1=[];
c2=[];
c3=[];


for i=1:length(y)
    
    if(y(i,classLabelIndex)==1)
        
        c1=[c1 ;y(i,1:num_features)];
        
    elseif(y(i,classLabelIndex)==2)
        
        c2=[c2 ;y(i,1:num_features)];
        
        
    elseif(y(i,classLabelIndex)==3)
        
        c3=[c3 ;y(i,1:num_features)];
        
        
    end
    
    
end

class1=c1(1:7500,1:num_features);
class2=c2(1:7500,1:num_features);
class3=c3(1:7500,1:num_features);



end