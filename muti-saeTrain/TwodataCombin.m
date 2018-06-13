function [ output_args ] = TwodataCombin( er1, er2 ,map1, map2)
%   er1, er2 合并
%   此处显示详细说明
   
   er_1       = [er1';map1];
   er_2       = [er2';map2];
   er_combin  = [er_1 er_2];
   OutEr      =  sortrows(er_combin', 2 );          % 27万 * 2 排序
   outEr = reshape(OutEr(:,1),1540,1741);
   save(strcat('E://DataResulterGraph/3-29-2类数据训练图/threelays/20-17聚类后合并.mat'));
end

