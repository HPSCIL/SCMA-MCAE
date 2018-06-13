%   分三类训练，最后结果合并     


load('D:\Program Files\MATLAB\R2014a\toolbox\final-DeepLearnToolbox-master\datadeal\1540data_clafy.mat');
     map_x1 = map_1(1:5,:)';
     map_x2 = map_2(1:5,:)';
     map_x3 = map_3(1:5,:)';
     
     t1 = threelays(map_x1,1540,1741,1,14797);      
     t2 = threelays(map_x2,1540,1741,2,36979);
     t3 = threelays(map_x3,1540,1741,3,14863);
% load('E://DataResulterGraph/分类计算训练图/threelays/神经元数据库190个神经元1千.mat.mat');
% load('D:\Program Files\MATLAB\R2014a\toolbox\final-DeepLearnToolbox-master\1540data_clafy.mat');
% load('D:\Program Files\MATLAB\R2014a\toolbox\final-DeepLearnToolbox-master\1540data_clafy.mat');

%      t1 = threelays(train_x_six,1540,1741);  
%      t2 = fivelays(train_x_six,190,250,283); 
%      t3 = sevenlays(train_x_six,t2,250,283);  
%      t4 = ninelays(train_x_six,t3,1540,1741); 
%      t5 = elevenlays(train_x_six,t4,250,283);  
% 
%      disp(['11层网络结构为:5-',num2str(t5(1,1)), '-' ,num2str(t5(1,2)), '-',num2str(t5(1,3)), '-',...
%          num2str(t5(1,4)),'-5-',num2str(t5(1,4)), '-' ,num2str(t5(1,3)), '-',num2str(t5(1,2)), '-',num2str(t5(1,1)),'-5']);
%      disp(['第十一层最大误差神经元数是：', num2str(t5(1,5))]);
     
%  figure;plot(er);
%  saveas(gcf,strcat('E://DataResulterGraph/整体网络训练图/网络不同层数误差变化图'),'fig');