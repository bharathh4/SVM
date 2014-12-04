function predict = combineBinaryDecisons(p12,p23,p13)
% Combines binary decision to create multi class SVM from 2-class SVM
predict=mode([p12 p23 p13]);

end