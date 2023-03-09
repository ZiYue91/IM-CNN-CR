function  dH  = caldH(ori,proc)

HSV_ori=rgb2hsv(ori);
H_ori=HSV_ori(:,:,1);

HSV_proc=rgb2hsv(proc);
H_proc=HSV_proc(:,:,1);

dH=abs(H_proc-H_ori);


