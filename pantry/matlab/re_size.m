function out = re_size (imgs, size)
    
    imgs = reshape(imgs, [size(imgs,1),sqrt(size(imgs,2)),sqrt(size(imgs,2)),3]);
    out = zeros(size);
    
    for i = 1:size(imgs,1)
       out(i,:,:) = rgb2gray(squeeze(imgs(i,:,:,:)));
    end
end
    