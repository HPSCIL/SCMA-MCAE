function [ output_args ] = allTrain_Parfor( input_args )
%ALLTRAIN 此处显示有关此函数的摘要
%   此处显示详细说明

 load('D:\Program Files\MATLAB\R2014a\toolbox\final-DeepLearnToolbox-master\data\AElatdata\T_sunshi''5-40-20-10-5''.mat');
 train_x_six = train_x_six;
%  er(1,1) = threelays(train_x_six,250,283);
% 
   tic;
%    parfor  i = 1 : 2
     spmd
         switch  labindex
             case  1
             threelays(train_x_six,250,283);
             case  2
             fivelays(train_x_six,250,283);
%              case(i == 3)
%              sevenlays(train_x_six,250,283);
%              case(i == 4)
%              ninelays(train_x_six,250,283);
%              case(i == 5)
%              elevenlays(train_x_six,250,283);
         end
   end
   t = toc;
   disp(['allTrain_Parfor : ' ' Took ' num2str(t) ' seconds']);
%  figure;plot(er);
%  saveas(gcf,strcat('E://DataResulterGraph/整体网络训练图/网络不同层数误差变化图'),'fig');
end

