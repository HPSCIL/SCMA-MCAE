function finalOut = nnfse(nn, x , batNum)   %%% 计算前向网络输出
%
%  x 太大的时候，要做切分
%  不保留 网络中间层的输出 a{i}
%  
%

    j           =  1;                              %分为几个包
    dataNum     =  batNum;                           %以70000作为批次，进行前向计算
    n           =  nn.n;                           %网络层数    
    surplusNum  =  mod( size(x, 1),  dataNum);     %余数
    inputNum    =  fix( size(x, 1)/dataNum);       %商
    nn.testing  =  1;                                %进入testing
      
% %  如果输入数据x过大,且除以dataNum有余数，将数组切分。并讲余数以0补齐，形成行数为dataNum的矩阵输入数据
%     if ( size(x,1)  >=  dataNum  )
%         
%         m           =  size(x, 1);
%         inputNum    =  fix( size(x, 1)/dataNum);
%                 
%         for  i = 1 : 1 :  inputNum 
%             eval(['sub_x_' num2str(i) '=' 'x(' num2str((i-1) * dataNum +1) ':'  num2str((i) * dataNum) ',:);' ]) ; % 获得变化的变量名
% %             eval(['' num2str(i) '=' 'y(' num2str((i-1) * dataNum +1) ':'  num2str((i) * dataNum) ',:);' ]) ; % 获得变化的变量名
%             j = j + 1;
%         end
%         
%         if ( surplusNum ~= 0 )
%             eval(['sub_x_' num2str(j) '=' 'x(' num2str( m - surplusNum + 1) ':'  num2str(m) ',:);' ]) ;
%             eval(['temp = '  'zeros( '  num2str( dataNum - surplusNum )  ', '  'size(x,2));']) ;                   % 生成替补矩阵0 
%             eval(['sub_x_', num2str(j) ,'= [ sub_x_' , num2str(j) , '; temp ];' ]) ;
%             %给y补充数字矩阵
%             eval(['y = [ x ; temp ];']) ;
%         else
%             j = j -1;
%         end
% 
%     else
%          sub_x_1 = x ;
%     end
 
    
% z为x 进行分包后的 包号
for z = 1 : 1 : inputNum + 1
    
%   eval([ 'm = size(sub_x_',  num2str(z), ', 1);']);     %多少个样本,70750
    
    if(z ~= inputNum +1)
        eval(['sub_x_', num2str(z), ' = x((z - 1) * dataNum + 1 : z * dataNum, :);']);   %
    else
        eval(['sub_x_', num2str(z), ' = x((inputNum) * dataNum + 1 : size(x, 1), :);']); 
        eval(['temp = '  'zeros( '  num2str( dataNum - surplusNum )  ', '  'size(x,2));']) ;                   % 生成替补矩阵0 
        eval(['sub_x_', num2str(z) ,'= [ sub_x_' , num2str(z) , '; temp ];' ]) ;
    end
       
    eval(['sub_x_', num2str(z), ' = [ones(dataNum,1)',  '  sub_x_', num2str(z), '];' ]);
    eval(['nn.a{1} = sub_x_', num2str(z),';']);

    %remove bias term
    nn = nnff(nn, nn.a{1}(:,2:end), nn.a{1}(:,2:end));
    
    
    %   合并最后输出
    if   ~eval(['exist(''sub_a_',  num2str(n),''', ''var'')'])
        eval([ 'sub_a_',  num2str(n), ' = nn.a{n}; ']);
    else
        eval([ 'sub_a_',  num2str(n), ' = [sub_a_', num2str(n),'; nn.a{n}] ;']);
    end;
end

    eval([ 'finalOut = [sub_a_', num2str(n),'(1:size(x,1),:)];']);
    
    nn.testing = 0;              %退出testing
    
%     sub_a_n = sub_a_n(size(x),:);
%  
% %   将每一层的a输出都算一遍
% for p = 2 : 1 : nn.n
%      eval([ 'nn.a{',  num2str(p), '} = [sub_a_', num2str(p), '];']);    
% end


 end