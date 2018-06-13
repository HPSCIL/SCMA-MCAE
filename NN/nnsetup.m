function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'PreLu';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 0.2;            %  learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.scaling_learningRate             = 1;            %  0.999Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0.00001;            %  L2 regularization
    nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0.1;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
    nn.sigmParmA                        = 1;            %  sigm中的倾斜参数
    nn.sigmParmB                        = 0;            %  sigm中中轴参数



%        switch z 
%             case 1     %第一类firstWeight1-1-filter.mat  预训练sae中的w
%                 load('E:\DataResulterGraph\dropTest\if_filter_compare\firstWeight1-1-filter.mat', 'nn_');
%                 nn = nn_;
%             case 2     %firstWeight1-2-filter.mat  fine训练nn中的w
%                 load('E:\DataResulterGraph\dropTest\if_filter_compare\firstWeight1-2-filter.mat', 'nn_');
%                 nn = nn_;
%             case 3     %第二类firstWeight2-1-filter.mat  预训练sae中的w
%                 load('E:\DataResulterGraph\dropTest\if_filter_compare\firstWeight2-1-filter.mat', 'nn_');   
%                 nn = nn_;
%             case 4     %第二类firstWeight2-2-filter.mat  预训练sae中的w
%                 load('E:\DataResulterGraph\dropTest\if_filter_compare\firstWeight2-2-filter.mat', 'nn_');   
%                 nn = nn_;
%        end;
        nn.alfa(1)  = 1;
    for i = 2 : nn.n       
      %  weights and weight momentum
        nn.W{i - 1}  = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
      %  average activations (for use with sparsity)
        nn.p{i}      = zeros(1, nn.size(i));   
        save(strcat('E:\DataResulterGraph\dropTest\if_filter_compare\firstWeight.mat')); 
     
      %  for PReLU
        nn.alfa(i) = 3 + ( 8 -3 ).* rand(1);           % 生成U[3,8]的随机数
        
    end
end
