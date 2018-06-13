function [er, bad] = nntest(nn, x, y)
    % er 是不期望的值占总数的多少比例；bad是位置
    % labels 为位置指标array
    
    labels = nnpredict(nn, x);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
    
end
