function [ sae ] = saeRawTrainProgram( sae, x, opts )
%SAERAWPROGRAM 原始sae训练程序，去掉了 可视化7075个batch的误差值变化 以及 每增加一层，平均误差的变化情况

    n = numel(sae.ae);        %sae的个数
    inputSize = size(x,1);
    for i = 1 : numel(sae.ae);
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
        [sae.ae{i}, l] = nntrain(sae.ae{i}, x, x, opts);
        %         Llay{i,1} = l;        %  是每个batch的误差率,一共7075个batch
        
        t = nnff(sae.ae{i}, x, x);
        %         Llayer(i,1) = t.L;    %  用于记录每增加一层，平均误差的变化情况
        x = t.a{2};
        %remove bias term
        x = x(:,2:end);
     
    end

end

