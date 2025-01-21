%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Adaptive step-size iteration (PIE with adpative alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for num = 1 : LED_num_x*LED_num_y
    
    % Calcute the step-size for the next iteration
    if(num ==1 && iter==1)
        Alpha = 1;
        Err_bef = inf;
    elseif(num ==1 && iter>1)
        eval Calc_stepsize;
    end
    
    % Get the subspecturm
    kx = round(kxky_index(num,1));
    ky = round(kxky_index(num,2));
    Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
    Abbr_Subspecturm = Subspecturm.*Aperture_fun;
    
    % Real space modulus constraint
    Uold = ifft2(fftshift(Abbr_Subspecturm));
    Unew = RAW(:,:,Image_num_index(num)).*(Uold./abs(Uold));
    
    % Fourier space constraint and object function update
    Abbr_Subspecturm_corrected = fftshift(fft2(Unew));
    Subspecturm = F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1);
    
    W = Alpha*abs(Aperture_fun)./max(max(abs(Aperture_fun)));
    
    invP = conj(Aperture_fun)./((abs(Aperture_fun)).^2+eps.^2);
    Subspecturmnew = (W.*Abbr_Subspecturm_corrected + (1-W).*(Abbr_Subspecturm)).*invP;
    Subspecturmnew(Aperture==0) = Subspecturm(Aperture==0);
    
    % Fourier sperturm replacement
    F(Fcenter_Y+ky-fix(M/2): Fcenter_Y+ky+ceil(M/2)-1, Fcenter_X+kx-fix(N/2):Fcenter_X+kx+ceil(M/2)-1) = Subspecturmnew;
    
    % Inverse FT to get the reconstruction
    Result = ifft2(fftshift(F));
end

