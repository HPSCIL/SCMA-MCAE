function [ output_args ] = filter( input_args )
%FILTER �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��


%%%%��ֵ�˲�
 medfilter1 = medfilt2(idx ,[10,10]); figure;contourf( medfilter1);
figure;contourf( flipud(medfilter1Copy));

%%%%   �˲��������࣬��������40��
w     =  fspecial('disk', 5);
kmeans_filter     =  imfilter(kmeans1,w,'replicate');
kmeans_filter(kmeans_filter>1.5)=2;kmeans_filter(kmeans_filter<=1.5)=1;figure;visualize(kmeans_filter);

% �ֶ��ٴ���filter֮��ĵ�
kmeans_filter(1:50,250:283)=2;visualize(kmeans_filter);

%%%%%   >>>>>>>>
 aeOut_filter(3.2>aeOut_filter & aeOut_filter>2.5)=3;aeOut_filter(aeOut_filter>3.2)=4;visualize(aeOut_filter);
 aeOut_filter(2.5>aeOut_filter & aeOut_filter>1.5)=2;aeOut_filter(aeOut_filter<=1.5)=1;visualize(aeOut_filter);

 %��filter�õ�������ԭʼ������ϵ����
 train_x_filter = [train_x_six';reshape(x,1,70750)]
 visualize(aeOut_filter);

contourf(flipud(rawoutEr_log))
end
