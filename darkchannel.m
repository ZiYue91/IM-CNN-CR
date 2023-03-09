function p = darkchannel(Image,kernelsize)
[height,width,~] = size(Image);
DC = zeros(height,width);
for y=1:height
    for x=1:width
        DC(y,x) = min(Image(y,x,:));
    end
end
Darkchannel = minfilt2(DC, [kernelsize,kernelsize]) ;
p = Darkchannel;
end