function DC = calDC(DehazeImage)

DehazeImage = im2double(DehazeImage);
kernelsize = 5; 
Darkchannel = darkchannel(DehazeImage,kernelsize);
DC = Darkchannel;
