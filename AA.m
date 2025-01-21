%% Adaptive alpha strategy  (with alpha =1.9)


   for num = 1 : LED_num_x*LED_num_y  % idxUsed
%      Get the subspecturm      
       kx = round(kxky_index(num,1));
       ky = round(kxky_index(num,2));
       Subspecturm1 = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
       Abbr_Subspecturm1 = Subspecturm1.*Aperture_fun;     
    
       % Real space modulus constraint  
       Uold1 = ifft2(fftshift(Abbr_Subspecturm1));     
       Unew1 = RAW1(:,:,Image_num_index(num)).*(Uold1./abs(Uold1));  

       Threshold(num,iter)=real(mean(mean(RAW1(:,:,Image_num_index(num))-Unew1)));

       if iter==1
            RAW2(:, :, Image_num_index(num)) = RAW1(:,:, Image_num_index(num))-mean(Threshold(num, iter));
            alpha=2*max(max(abs(Aperture_fun)))-mean(Threshold(:,iter));   
       else
            RAW2=RAW1;
       end

       Unew1 = RAW2(:,:,Image_num_index(num)).*(Uold1./abs(Uold1));

 
    % Fourier space constraint and object function update
       Abbr_Subspecturm_corrected1 = fftshift(fft2(Unew1));   
       W1 =1*abs(Aperture_fun)./max(max(abs(Aperture_fun)));    

       invP1 = conj(Aperture_fun)./((abs(Aperture_fun)).^2+eps^2);
       Subspecturmnew1 = Abbr_Subspecturm_corrected1 + (alpha-W1).*(Abbr_Subspecturm1).*invP1;
       Subspecturmnew1(Aperture==0) = Subspecturm1(Aperture==0);

      % Fourier sperturm replacement
       F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1) = Subspecturmnew1;
            
       % Inverse FT to get the reconstruction
       Result(:,:,iter) = ifft2(fftshift(F));  

   end    
      
