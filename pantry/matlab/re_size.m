function out = re_size (imgs, shape)
    
    imgs = reshape(imgs, [size(imgs,1),sqrt(size(imgs,2)/3),sqrt(size(imgs,2)/3),3]);
    out = zeros([size(imgs,1),shape]);
    
    for i = 1:size(imgs,1)
       out(i,:,:) = imresize(rgb2gray(uint8(squeeze(imgs(i,:,:,:)))),[28,28]);
    end
end
    