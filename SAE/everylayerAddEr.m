function everylayerAddEr
%EVERYLAYERADDER 用于计算每层中，每次增加神经元，误差函数以及聚类变化情况
%%  输入数据为9层训练的CG数据，5-40-20-10-5-10-20-40-5
%%  参数声明与赋值
interNum = 200 ;               %迭代次数
batchSizeNum = 10 ;            %batch的大小，必须要能被样本总数整除
saeLearningRate = 0.3;         %预训练部分的学习率
AllNNLearningRate = 0.8;       %整体网络
NeuralNumsinglelay = 10;       %隐含单层神经元的数目,起始数
endNeuralNum = 200;            %隐含单层神经元的数目,终止数
AddNerualNum = 10 ;            %每次增加多少个神经元

% startendNum = 5 ;              %头尾神经元数一样
loadAddress = 'D:\Program Files\MATLAB\R2014a\toolbox\final-DeepLearnToolbox-master\data\AElatdata\T_sunshi''5-40-20-10-5''.mat';  %需要加载的数据文件位置
% saveAddress = 'D:\Program Files\MATLAB\R2014a\toolbox\final-DeepLearnToolbox-master\data\everylayerAddEr\everylayerAddEr.mat';  %需要存储的数据文件位置
% everLayNeurNumAddAddr = strcat('D:\Program Files\MATLAB\R2014a\toolbox\final-DeepLearnToolbox-master\data\everylay',num2str(startendNum) ,'.mat');  %需要存储的数据文件位置
%%%%%%%%%%%%%%%%%%%%%%%
load(loadAddress);
rand('state',0)   

   
    for i = NeuralNumsinglelay : AddNerualNum : endNeuralNum
            
            Saestruct = [5  i];        %SAE网络初始层设置
            sae = saesetup(Saestruct);
            sae.ae{1}.activation_function       = 'sigm';
            sae.ae{1}.learningRate              = saeLearningRate;
            opts.numepochs =  interNum;
            opts.batchsize = batchSizeNum;
            [sae] = saeRawTrainProgram(sae, x, opts);
          
            % encoder的w的转置+decoder的b。
            wae1 = [sae.ae{1}.W{2}(:,1)';sae.ae{1}.W{1}(:,2:end)]; 

            % Use the SDAE to initialize a FFNN  fine_tuning过程
            nn = nnsetup([startendNum  i  startendNum]);
            nn.activation_function              = 'sigm';
            nn.learningRate                     = AllNNLearningRate;

            nn.W{1} = sae.ae{1}.W{1};                        %  encoder层采用w1的权值。discoder用w1+w2的偏执
            nn.W{2} = wae1';
            nn.output              = 'sigm';                  %  use softmax output

            %%%%%%%%%%% Train the FFNN
            opts.numepochs =   interNum;
            opts.batchsize =  batchSizeNum;
            [nn,lfull] = nntrain(nn, x, x, opts);
            
            figure;plot( lfull);   %迭代次数下的误差
            saveas(gcf,strcat('E://DataResulterGraph/整体网络训练图/', numel(sae.ae), '层整体网络迭代误差值'),'fig');
            
            subCG = nnfse(nn, x);                      %输出每层的数据
            
            %%%%%%%%%%% 计算重构与原始数据差异，可视化异常结果
            rawoutEr =  computerRawAndRestroeL( subCG{1,1}, train_x_six);     %每增加一次神经元数，生成的数据都与原始数据进行比较
            Aller(j,1) = mean(rawoutEr(:));                       %所有样本总误差计算
            rawoutEr =  reshape(rawoutEr,250,283);
            figure;contourf(flipud(rawoutEr)); 
            axis off;               %让轴不可见；
%           title(strcat(num2str(i) ,'个神经元异常分布'));
            saveas(gcf,strcat('E://DataResulterGraph/单层训练结果图/',num2str(z),'层',num2str(i),'个特征的异常分布'),'fig')  %如果只有一幅图，handle设为gcf           
            close(figure(gcf));            
            
            %%%%%%%%%%   记录每增加神经元数，聚类效果的变化
            disp([num2str(i) '个特征数，输出聚类']); 
            opts = statset('Display','final','MaxIter',1000);   %数据大了，100次不收敛，改为1000次
            idx = kmeans(subCG{1,1},5,'Options',opts);             
            idx = reshape(idx,250,283);
            idx = flipud(idx);
            figure;contourf(idx); 
            axis off;               %让轴不可见；
%           title(strcat('第',num2str(i), '层聚类效果'));
            saveas(gcf,strcat('E://DataResulterGraph/单层训练结果图/第',num2str(z),'层',num2str(i),'个特征的聚类效果'),'fig') 
            close(figure(gcf));     %关闭以上生成的图
            
            j = j+1 ;           %用于记录误差值的位置
    end
    
        figure;plot(Aller );
        if (size(Aller(:,1)) == 20)
            set(gca,'xtick',[10:10:200]);
%             set(gca,'xticklabel',{'2005年','2006年','2007年','2008年','2009年','2010年'});
        else
            set(gca,'xtick',[5:5:40]);
        end
    %     title(strcat('第',num2str(z),'层，每增加10个神经元，误差函数的变化情况'));
        saveas(gcf,strcat('E://DataResulterGraph/单层训练结果图/第',num2str(z),'层误差函数的变化情况'),'fig') 
        close(figure(gcf));   
        disp('*****');
        
end

