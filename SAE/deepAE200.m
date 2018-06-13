function deepAE200
%DEEPAE200 Summary of this function goes here
%   Detailed explanation goes here
%###############################  9层网络
%########################################################
 %%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
load('d:\Program Files\MATLAB\R2012a\toolbox\final-DeepLearnToolbox-master\data\MAPminmax.mat')

rand('state',0)
sae = saesetup([5 40 20 10 5]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.;
sae.ae{2}.activation_function       = 'sigm';
sae.ae{2}.learningRate              = 1;
sae.ae{2}.inputZeroMaskedFraction   = 0.;
sae.ae{3}.activation_function       = 'sigm';
sae.ae{3}.learningRate              = 1;
sae.ae{3}.inputZeroMaskedFraction   = 0.;
sae.ae{4}.activation_function       = 'sigm';
sae.ae{4}.learningRate              = 1;
sae.ae{4}.inputZeroMaskedFraction   = 0.;
opts.numepochs =  200;
opts.batchsize = 10;
sae = saetrain(sae, train_x_six, opts);

% visualize(sae.ae{1}.W{1}(:,2:end)')  %可视化权重
% figure; visualize(sae.ae{1}.W{1}');
% figure; visualize(sae.ae{2}.W{1}');
% figure; visualize(sae.ae{3}.W{1}');
% figure; visualize(sae.ae{4}.W{1}');



% encoder的w的转置+decoder的b。
wae1 = [sae.ae{4}.W{2}(:,1)';sae.ae{4}.W{1}(:,2:end)]; 
wae2 = [sae.ae{3}.W{2}(:,1)';sae.ae{3}.W{1}(:,2:end)];
wae3 = [sae.ae{2}.W{2}(:,1)';sae.ae{2}.W{1}(:,2:end)];
wae4 = [sae.ae{1}.W{2}(:,1)';sae.ae{1}.W{1}(:,2:end)];


% Use the SDAE to initialize a FFNN  fine_tuning过程
nn = nnsetup([5 40 20 10 5 10 20 40 5]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.W{3} = sae.ae{3}.W{1};
nn.W{4} = sae.ae{4}.W{1};
nn.W{5} = wae1';   
nn.W{6} = wae2';
nn.W{7} = wae3';
nn.W{8} = wae4';

nn.output              = 'sigm';    %  use softmax output
CG = nnfse(nn, train_x_six);
cn = elcomputer(train_x_six,CG);
CJELDeepAE = reshape(cn,250,283);
figure; visualize(CJEL1);
dlmwrite('test.txt',CJEL1);
disp('*****');  
save('d:\Program Files\MATLAB\R2012a\toolbox\final-DeepLearnToolbox-master\data\eightSAE200.mat');

end

