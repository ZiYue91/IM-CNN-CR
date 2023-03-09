clear
clc
name='test1115';

image_dir  =  ['./results/' name '/test_latest/images/'];
im_dir  = dir([image_dir '*.png']);

eachnum=12;
num = length(im_dir)/eachnum;

result_simu=zeros(500,3);
result_real_DC=zeros(350,2);
result_real_dH=zeros(350,2);
simu_idx = 0;
real_idx = 0;

for i = 1:1:num
    thicknessmap=imread([image_dir, im_dir(eachnum*(i-1)+12).name]);
    
    ori543=imread([image_dir, im_dir(eachnum*(i-1)+5).name]);
    ori321=imread([image_dir, im_dir(eachnum*(i-1)+4).name]);
    ori = cat(3,ori543(:,:,2:3),ori321(:,:,2));
    
    proc543=imread([image_dir, im_dir(eachnum*(i-1)+7).name]);
    proc321=imread([image_dir, im_dir(eachnum*(i-1)+6).name]);
    proc = cat(3,proc543(:,:,2:3),proc321(:,:,2));
    
    gt543=imread([image_dir, im_dir(eachnum*(i-1)+1).name]);
    gt321=imread([image_dir, im_dir(eachnum*(i-1)+2).name]);
    gt = cat(3,gt543(:,:,2:3),gt321(:,:,2));
    
    thicknessmap=im2double(thicknessmap(:,:,1));
    ori=im2double(ori);
    proc=im2double(proc);
    gt=im2double(gt);
   
    if strfind(im_dir(eachnum*(i-1)+12).name,'real')
       real_idx = real_idx+1;
       
       dH = caldH(ori,proc);
       DC = calDC(proc);
        
       cloud = double(im2bw(thicknessmap,0.15)); 
       noncloud = 1-cloud;

       if sum(sum(cloud))/(256*256)>0.01
           result_real_DC(real_idx,1) = sum(sum(cloud));
           result_real_DC(real_idx,2) = sum(sum(DC.*cloud));
       end

       if sum(sum(noncloud))/(256*256)>0.01
           result_real_dH(real_idx,1) = sum(sum(noncloud));
           result_real_dH(real_idx,2) = sum(sum(dH.*noncloud));
       end
    else
        simu_idx = simu_idx+1;
        result_simu(simu_idx,1)=immse(proc,gt);
        result_simu(simu_idx,2)=psnr(proc,gt);
        result_simu(simu_idx,3)=ssim(proc,gt);
    end
    
    disp(i)
end

result_simu_mean = mean(abs(result_simu))
result_real_DC_mean = sum(result_real_DC(:,2))./(sum(result_real_DC(:,1)))
result_real_dH_mean = sum(result_real_dH(:,2))./(sum(result_real_dH(:,1)))




