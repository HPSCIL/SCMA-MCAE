function [ rawoutEr ] = computerRawAndRestroeL( x, y )
%COMPUTERRAWANDRESTROEL 计算原始数据与重构数据的之差
%   将原始数据中的属性平方和，并开方。（数据要求：行为样本数，列为属性数。）
%   将重构数据中的属性平方和，并开方。
%   将两者结果求差值

    outdot = dot(x',x');                    %求输出的点积 
    outdot = sqrt(outdot');                             %每个样本开方,得到样本数*1的矩阵
    rawdatadot = dot(y',y');        %计算原始数据的内积
    rawdatadot = sqrt(rawdatadot');  
    rawoutEr =  imsubtract(outdot,rawdatadot);          %求原始样本与输出样本之差

end

