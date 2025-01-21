%% Initialize HR image
% Upsample the central low-resolution image


for m = 1:LED_num_x*LED_num_y
    m;                               
    bk1 = mean2(double(RAW(35:45,40:50,m)));     
    bk2 = mean2(double(RAW(82:92,75:85,m)));    
    Ibk(m) = mean([bk1,bk2]);
    
    if Ibk(m)>300
        Ibk(m) = Ibk(m-1);
    end         
end

Ithresh_reorder = RAW;
Ibk_reorder = Ibk;


IbkThr1=0.4;
% denoise
for m = 1:LED_num_x*LED_num_y
    SSSS(:,m)=mean(mean(RAW(:,:,m)));    
    Itmp = Ithresh_reorder(:,:,m);    
    if SSSS <= IbkThr1
         Itmp1 = Itmp-0.10;       
    else
         Itmp1 = Itmp-0.03;   
       
    Itmp1(Itmp1<0) = 0;
    Ithresh_reorder(:,:,m) = Itmp1;  
end
end
RAW1 = Ithresh_reorder;

[a,b] =  find (SSSS > IbkThr1);   
[c,d] =  find (SSSS <= IbkThr1);  


A = imresize(RAW1(:,:,LED_center),Mag_image); %RAW

[Hi_res_M,Hi_res_N] = size(A);
% Initialize the HR image using the amplitude of the central low-resolution image.
F = fftshift(fft2(A));
Fcenter_X = fix(Hi_res_M/2)+1;
Fcenter_Y = fix(Hi_res_N/2)+1;