function [ output_args ] = k-means(nnOutdata1, n )
%K-MEANS kmeans 将数据分为n类
%  nnOutdata1 为神经网络m个样本，p个属性值

%%%%聚类的代码：
opts = statset('Display','final','MaxIter',1000); 
kmeans112 = kmeans( nnOutdata1 , n ,'Options',opts);

idx = reshape(kmeans112,250,283);
            idx = flipud(idx);
            figure;contourf(idx);

end

