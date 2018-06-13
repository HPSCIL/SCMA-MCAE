function rawoutEr = fivelays(train_x_six,NeuralNum,m,batNum)
%%  [5 200 i]
%%  此处显示详细说明



iterNum = 2000 ;              %迭代次数
batchSizeNum =  batNum;       %batch的大小，必须要能被样本总数整除：70750个数据：1415
saeLearningRate = 1;        %预训练部分的学习率
NNLearnRate = 0.8;            %整体网络
startNeuralNum = NeuralNum;          %隐含单层神经元的数目,起始数
endNeuralNum = 100;           %隐含单层神经元的数目,终止数 （400）
AddNerualNum = 10 ;           %每次增加多少个神经元
tm = 200;
filename = ['E:\0nanhuashan\多层训练\fivelays\',num2str(batNum),'\'];
if ~exist(filename)
    new_folder = filename;
    mkdir(new_folder);
end
rand('state',0)   

j = 1;
threelayEr = 0;

disp('***********************************五层网络训练*********************************************');
     for i = startNeuralNum : AddNerualNum : endNeuralNum
            
            Saestruct = [4 tm i];        %SAE网络初始层设置
            sae = saesetup(Saestruct);
            sae.ae{1}.activation_function       = 'PreLu';
            sae.ae{1}.learningRate              = saeLearningRate;
% %             sae.ae{1}.dropoutFraction = 0.8;         %dropout随机舍弃50%的隐含层
%             sae.ae{2}.activation_function       = 'PreLu';
%             sae.ae{2}.learningRate              = saeLearningRate;
% %             sae.ae{2}.dropoutFraction = 0.7;         %dropout随机舍弃50%的隐含层
            
            opts.numepochs =  iterNum;
            opts.batchsize = batchSizeNum;
            [sae] = saeRawTrainProgram(sae, train_x_six, opts);
            
            % encoder的w的转置+decoder的b。
            wae1 = [sae.ae{2}.W{2}(:,1)';sae.ae{2}.W{1}(:,2:end)]; 
            wae2 = [sae.ae{1}.W{2}(:,1)';sae.ae{1}.W{1}(:,2:end)]; 
        

            % Use the SDAE to initialize a FFNN  fine_tuning过程
            nn = nnsetup([4 tm i tm 4]);
            nn.activation_function              = 'PreLu';
            nn.learningRate                     = NNLearnRate;

            nn.W{1} = sae.ae{1}.W{1};                        %  encoder层采用w1的权值。discoder用w1+w2的偏执
            nn.W{2} = sae.ae{2}.W{1}; 
            nn.W{3} = wae1';
            nn.W{4} = wae2';
                  
            nn.output              = 'sigm';                  %  use softmax output

            %%%%%%%%%%% Train the FFNN
            opts.numepochs =   iterNum;
            opts.batchsize =  batchSizeNum;
            [nn,lfull] = nntrain(nn, train_x_six, train_x_six, opts);
            layLfull(j) = lfull(end);                                      %整个样本均值
            NNoutdata = nnfse(nn, train_x_six,batNum);                      %输出每层的数据
            
%             %%%5-40-20-10-5%  第四十个的时候，整体网络误差值
%             if(i == 20)
%                 n = size(lfull,1);
%                 threelayEr = lfull(n,1);
%             end
            
            figure;plot(lfull);   %迭代次数下的误差
            saveas(gcf,strcat(filename , num2str(i),'个神经元整体网络迭代误差值'),'fig');
            close(figure(gcf));     %关闭以上生成的图
% %                         
   %%%%%%%%%%% 计算重构与原始数据差异，可视化异常结果
            nnOutdata1 =  gather(NNoutdata); 
            rawoutEr =   elcomputer( train_x_six, nnOutdata1);   %重构误差
%             rawoutEr_log = log(rawoutEr);     %取对数看清楚关系                        
%             rawoutEr_log =  reshape(rawoutEr_log,247,279);          %250,283
%             figure;contourf(flipud(rawoutEr_log)); 
%             axis off;              %让轴不可见；
%             title(strcat(num2str(i) ,'个神经元异常分布'));
%             saveas(gcf,strcat(filename, num2str(i),'个特征的异常分布1千'),'fig')  %如果只有一幅图，handle设为gcf           
%             close(figure(gcf));          
            
            %马氏距离
            mahal_d =   mahal( nnOutdata1, train_x_six);
%             mahal_d =   reshape(mahal_d,247,279); 
            mahal_d_log = log(mahal_d);     %取对数看清楚关系
% %             figure;contourf(flipud(mahal_d_log));
% %             axis off;              %让轴不可见；
% %             title(strcat(num2str(i) ,'个神经元马氏距离异常分布'));
% %             saveas(gcf,strcat(filename, num2str(i),'个特征的马氏距离异常分布'),'fig')  %如果只有一幅图，handle设为gcf           
% %             close(figure(gcf));            
            save(strcat(filename,'prelu-',num2str(i),'个神经元.mat'));       
                        
%             %%%%%%%%%%   记录每增加神经元数，聚类效果的变化
%             disp(['五层网络，',num2str(i), '个特征数，输出聚类']); 
%             opts = statset('Display','final','MaxIter',1000);   %数据大了，100次不收敛，改为1000次
%             nnoutdata2   =   gather(NNoutdata{2,1});
%             idx = kmeans(nnoutdata2, 5,'Options',opts);             
%             idx = reshape(idx,r,c);
%             idx = flipud(idx);
%             figure;contourf(idx); 
%             axis off;               %让轴不可见；
% %           title(strcat('第',num2str(i), '层聚类效果'));
%             saveas(gcf,strcat(filename ,num2str(i),'个特征的聚类效果'),'fig') 
%             close(figure(gcf));     %关闭以上生成的图
            
            j = j+1 ;           %用于记录误差值的位置
                       
    end
    
        figure;plot(layLfull);
%         
%        %找出最大的或最小的
%          max1 = max(layLfull);
%          mm   = max(max1);
%         [~,column] = find(layLfull==mm);
%         t(1,1) = tm;
%         t(1,2) =  startNeuralNum + (column-1) * AddNerualNum;
%        
        title(strcat('第',num2str(m),'层，每增加10个神经元，误差函数的变化情况'));
        saveas(gcf,strcat(filename ,'5层-神经元数增加误差函数的变化情况'),'fig') 
        close(figure(gcf));   
        disp('*****');

end
