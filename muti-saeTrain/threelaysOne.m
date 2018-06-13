function [rawoutEr] = threelaysOne(train_x_six,NeuralNum,m,batNum)
%% [5  i]
%%  r,c 为行列号
%   m 为文件序号
%   batNum为一个batch的个数

iterNum = 500 ;               %迭代次数
batchSizeNum =  batNum;       %batch的大小，必须要能被样本总数整除：70750个数据：1415
saeLearningRate = 0.3;        %预训练部分的学习率
NNLearnRate = 0.3;            %整体网络
startNeuralNum = NeuralNum;          %隐含单层神经元的数目,起始数
endNeuralNum = NeuralNum;     %隐含单层神经元的数目,终止数 （400）
AddNerualNum = 10 ;           %每次增加多少个神经元
r  = 1540;
c  = 1741;

filename = ['E://DataResulterGraph/3-29-2类数据训练图/threelays/', num2str(m) , '/'];

rand('state',0)   

j = 1;
threelayEr = 0;
disp('*****************************三层网络训练***********************************************'); 

    for i = startNeuralNum : AddNerualNum : endNeuralNum    %神经元个数增加
            
            Saestruct = [5  i];        %SAE网络初始层设置
            sae = saesetup(Saestruct);
            sae.ae{1}.activation_function       = 'sigm';
            sae.ae{1}.learningRate              = saeLearningRate;
            opts.numepochs =  iterNum;
            opts.batchsize = batchSizeNum;
            [sae] = saeRawTrainProgram(sae, train_x_six, opts);
            
            % encoder的w的转置+decoder的b。
            wae1 = [sae.ae{1}.W{2}(:,1)';sae.ae{1}.W{1}(:,2:end)]; 

            % Use the SDAE to initialize a FFNN  fine_tuning过程
            nn = nnsetup([5  i  5]);
            nn.activation_function              = 'sigm';
            nn.learningRate                     = NNLearnRate;

            nn.W{1} = sae.ae{1}.W{1};                        %  encoder层采用w1的权值。discoder用w1+w2的偏执
            nn.W{2} = wae1';
            nn.output              = 'sigm';                  %  use softmax output

            %%%%%%%%%%% Train the FFNN
            opts.numepochs =   iterNum;
            opts.batchsize =  batchSizeNum;
            [nn,lfull] = nntrain(nn, train_x_six, train_x_six, opts);
            layLfull(j) = mean(lfull);
            NNoutdata = nnfse(nn, train_x_six, batNum);                      %输出每层的数据
            
%           %%%5-40-20-10-5%  第四十个的时候，整体网络误差值
%             if(i ==40)
%                 threelayEr = lfull(size(lfull,1),1);
%             end
            
            figure;plot(lfull);   %迭代次数下的误差
%             saveas(gcf,strcat(filename, num2str(i),'个神经元整体网络迭代误差值1千'),'fig');
            close(figure(gcf));     %关闭以上生成的图
                        
            %%%%%%%%%%% 计算重构与原始数据差异，可视化异常结果
            nnOutdata1 =  gather(NNoutdata); 
            rawoutEr =   elcomputer( train_x_six, nnOutdata1);   %重构误差
% %             rawoutEr =  reshape(rawoutEr,r,c);
% %             figure;visualize(rawoutEr); 
% %             axis off;              %让轴不可见；
% % %           title(strcat(num2str(i) ,'个神经元异常分布'));
% %             saveas(gcf,strcat(filename, num2str(i),'个特征的异常分布1千'),'fig')  %如果只有一幅图，handle设为gcf           
% %             close(figure(gcf));            
                        
            %%%%%%%%%%   记录每增加神经元数，聚类效果的变化
% %             disp(['三层网络，',num2str(i), '个特征数，输出聚类']); 
% %             opts = statset('Display','final','MaxIter',1000);   %数据大了，100次不收敛，改为1000次
% %             nnOutdata2 =  gather(NNoutdata{1,1}); 
% %             idx = kmeans(nnOutdata2,5,'Options',opts);             
% %             idx = reshape(idx,r,c);
% %             idx = flipud(idx);
% %             figure;contourf(idx); 
% %             axis off;               %让轴不可见；
%           title(strcat('第',num2str(i), '层聚类效果'));
% %             saveas(gcf,strcat(filename,num2str(i),'个特征的聚类效果1千'),'fig') 
% %             close(figure(gcf));     %关闭以上生成的图
            
            j = j+1 ;           %用于记录误差值的位置
            
            save(strcat(filename,'km-aeOut学习率为03-03',num2str(NeuralNum),'个神经元.mat'));
            
    end
    
        figure;plot(layLfull);
    
% %     %   找出最大的或最小的layfull
% %          max1 = max(layLfull);
% %          mm = max( max1 );
% %         [~,column] = find(layLfull==mm);
% %         tm =  startNeuralNum + (column-1) * AddNerualNum;
        
    %   title(strcat('第',num2str(z),'层，每增加10个神经元，误差函数的变化情况'));
        saveas(gcf,strcat(filename,'km-aeOut误差函数的变化情况1千'),'fig') ;
        close(figure(gcf));   
        disp('*****');       

%       clearvars -except tm；
end


