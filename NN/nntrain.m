function [nn,  Lfull]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net 
%  Lfull记录每次迭代的误差值  
%输入数据train x train y 量太大，程序有切分训练
%  训练的时候，以batch（1000）为批次训练，舍弃部分零碎
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize  =  opts.batchsize;
numepochs  =  opts.numepochs;

numbatches =  fix( m / batchsize) ; %取整数
numyushu   =  mod(m, batchsize);  %余数

% assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*(numbatches+1),1); 

Lfull = zeros(numepochs,1);
n = 1;                                 % n为batch的个数 初始值

for i = 1 : numepochs
    tic;
        kk = randperm(m);             %产生随机数

%     kk1 = kk(1:(m-numyushu));
%     kk2 = kk((m-numyushu):m);
    
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);   %
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(Gbatch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);  % 前向神经网络，不参与参数调整，只为获取输出，以及误差值
        nn = nnbp(nn);                    % 反馈神经网络，为了更新权值，计算dW
        nn = nnapplygrads(nn);            % 根据dW,算出权值W的变化量和更新结果
        
                
        L = gather(nn.L); 
        L(n) = L;    %每一个batch的误差平均值，（n=总数/batchsize,）
        
        n = n + 1;
    end
     
    
%     if(numyushu ~= 0)    %%    余数如果不等于0
%         
%         for q = 1 : 1 :  numyushu
%              
%         end    
%     end
    
    t = toc;
    %%  full-batch 训练
    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);  %nneval检验神经网络表现
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    else
        loss = nneval(nn, loss, train_x, train_y);        
        Lfull(i) = loss.train.e(end);                             % 每一个epoch的所有样本的平均误差       
        str_perf = sprintf('; Full-batch train err = %f',  Lfull(i));   %full batch 后 计算所有样本的平均误差值
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
%     disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
      disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1))))]);
    
     %用于动态调整学习率
      nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    

end
end

