function  intxt( x )
%INTXT Summary of this function goes here
%   Detailed explanation goes here
fid=fopen('b.txt','wt');%写入文件路径
[m,n]=size(a);
 for i=1:1:m
    for j=1:1:n
       if j==n
         fprintf(fid,'%g\n',a(i,j));
       else
          fprintf(fid,'%g\t',a(i,j));
       end
    end
end
fclose(fid);

end

