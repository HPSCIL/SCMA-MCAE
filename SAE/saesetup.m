function sae = saesetup(size)
%  此处修改了nnsetup，加入了w权值
    for u = 2 : numel(size)
        sae.ae{u-1} = nnsetup([size(u-1) size(u) size(u-1)]);
    end
end
