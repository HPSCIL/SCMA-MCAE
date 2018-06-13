function [sae, Llay] = saetrain(sae, x, opts)
   

    n = numel(sae.ae);        %sae的个数
    for i = 1 : numel(sae.ae);
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
        [sae.ae{i}, l] = nntrain(sae.ae{i}, x, x, opts);
% (1)        Llay{i,1} = l;        %  是每个batch的误差率,一共7075个batch
        t = nnff(sae.ae{i}, x, x);
% (1)       Llayer(i,11) = t.L;    %  用于记录每增加一层，平均误差的变化情况
                      
        %将每层特征的输出值作为下一层的输入值
        x = t.a{2};
        
        %remove bias term
        x = x(:,2:end);                
        %         subplot(2,2,i);
% (1)       figure;  plot(l);              %   可视化7075个batch的误差值变化
% (1)         title(strcat('第',num2str(i), '层7075个batch的误差变化'));
%  (1)        saveas(gcf,strcat('E://DataResulterGraph/5-40-20-10-5-10-20-40-5+200iter/第',num2str(i), '层7075个batch的误差变化'),'fig') ;    
    end
% (1)        Llay{1,2} = Llayer;
% (1)        figure;plot(Llayer);
% (1)        title(strcat('每增加一层，平均误差的变化情况'));
% (1)        saveas(gcf,strcat('E://DataResulterGraph/5-40-20-10-5-10-20-40-5+200iter/每增加一层，平均误差的变化情况'),'fig') ;
end
