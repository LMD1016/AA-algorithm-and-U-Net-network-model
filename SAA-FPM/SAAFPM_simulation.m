%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main file to implement the SAA-FPM for Fourier ptychographic reconstruction algorithm 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc, clear, close all;
 
%% Load and prepare initial images 
I0 = imresize(im2double(imread('I0.bmp')),  [400 400]);    % Resize amplitude image 
P0 = imresize(im2double(imread('P0.bmp')),  [400 400]);    % Resize phase image 
X3 = sqrt(I0) .* exp(sqrt(-1) .* P0);                     % Create complex field 
[a, b] = size(X3);
 
%% Setup parameters for Fourier ptychography 
[M, N] = size(X3);
 
% LED array parameters 
LED_num_x = 5;          % Number of LEDs in x direction   5×5(accurate) or 3×3(Speed)
LED_num_y = 5;          % Number of LEDs in y direction 
LED_center = (LED_num_x * LED_num_y + 1) / 2;  % Center LED index 
 
% Optical parameters 
NA = 0.1;               % Numerical aperture 
Mag = 4;                % Magnification 
LED2stage = 90e3;       % Distance from LED array to sample 
LEDdelta = 2.5e3;       % LED spacing (2.5mm)
Pixel_size = 2.4e-6 / Mag;  % Pixel size at sample plane 
Lambda = 0.532e-6;      % Wavelength (532nm)
 
% Calculate frequency domain parameters 
k = 2 * pi / Lambda;
kmax = 1 / Lambda * NA;
 
% Image scaling parameters 
Mag_image = 1;                          % Image magnification factor 
Pixel_size_image = Pixel_size / Mag_image;
Pixel_size_image_freq = 1 / Pixel_size_image / (M * Mag_image);
kmax = kmax / Pixel_size_image_freq;    % Scaled cutoff frequency (~45.0481)
 
%% Create aperture mask (pupil function)
[x, y] = meshgrid((-fix(M/2):ceil(M/2)-1), (-fix(N/2):ceil(N/2)-1));
[~, R] = cart2pol(x, y);
Aperture = ~(R > kmax);         % Binary aperture mask 
Aperture_fun = double(Aperture);% Convert to double 
 
%% Initialize image acquisition sequence 
Image_num_index = zeros(1, LED_num_x * LED_num_y);  % LED illumination sequence 
kxky_index = zeros(LED_num_x * LED_num_y, 2);       % Spatial frequency offsets 
loop_x = (LED_num_x + 1) / 2;                       % Number of illumination rings 
 
% Generate spiral illumination sequence 
for loop = 1:loop_x 
    if loop == 1 
        % First ring (center LED only)
        num = 1;
        Image_num_index(num) = LED_center;
        last_index = Image_num_index(num);
        num = num + 1;
    else 
        % Subsequent rings (spiral pattern)
        Image_num_index(num) = last_index - 1;
        last_index = Image_num_index(num);
        num = num + 1;
        Image_num_index(num:num+loop*2-4) = (last_index-LED_num_x : -LED_num_x : last_index-LED_num_x-(loop*2-4)*LED_num_x);
        num = num + loop*2 - 4;
        last_index = Image_num_index(num);
        num = num + 1;
        Image_num_index(num:num+loop*2-3) = (last_index+1 : 1 : last_index+1+(loop*2-3));
        num = num + loop*2 - 3;
        last_index = Image_num_index(num);
        num = num + 1;
        Image_num_index(num:num+loop*2-3) = (last_index+LED_num_x : LED_num_x : last_index+LED_num_x+(loop*2-3)*LED_num_x);
        num = num + loop*2 - 3;
        last_index = Image_num_index(num);
        num = num + 1;
        Image_num_index(num:num+loop*2-3) = (last_index-1 : -1 : last_index-1-(loop*2-3));
        num = num + loop*2 - 3;
        last_index = Image_num_index(num);
        num = num + 1;
    end 
end 
 
%% Simulate LED array misalignment (position and rotation errors)     

% thetaa=10*(rand(1)-0.5);
% leddx=4000*(rand(1)-0.5);
% leddy=4000*(rand(1)-0.5);

thetaa = -2.395;    % Rotation error in degrees
leddx = -1255;      % X position error (μm)
leddy = 888;        % Y position error (μm)
 
%% Calculate theoretical spatial frequency shifts for each LED 
for i = 1:LED_num_x 
    for j = 1:LED_num_y 
        % Apply rotation transformation 
        m = [(i-(LED_num_x+1)/2), (j-(LED_num_y+1)/2)] * [cos(thetaa*pi/180); sin(thetaa*pi/180)];
        n = [(i-(LED_num_x+1)/2), (j-(LED_num_y+1)/2)] * [-sin(thetaa*pi/180); cos(thetaa*pi/180)];
        
        % Apply position offset 
        x_num = (m * LEDdelta + leddx);
        y_num = (n * LEDdelta + leddy);
        
        % Calculate spatial frequency shift 
        distance = sqrt(y_num.^2 + x_num.^2);
        theta1 = atan2(distance, LED2stage);
        kr = 1 / Lambda * sin(theta1);
        theta = atan2(y_num, x_num);
        
        % Store frequency shifts 
        kxky_index0(i,j,2) = (kr * sin(theta) / Pixel_size_image_freq);
        kxky_index0(i,j,1) = (kr * cos(theta) / Pixel_size_image_freq);
    end 
end 
 
% Reorder frequency shifts according to illumination sequence 
kx_index = kxky_index0(:,:,1);
ky_index = kxky_index0(:,:,2);
kxky_index_0(:,1) = kx_index(Image_num_index(:));
kxky_index_0(:,2) = ky_index(Image_num_index(:));
 
%% Simulate low-resolution measurements 
numImg1 = LED_num_x * LED_num_y;
idxUsed = numImg1;   % Use all images 
X3 = imresize(X3, Mag_image);
[Hi_res_M, Hi_res_N] = size(X3);
Fcenter_X = fix(Hi_res_M/2) + 1;
Fcenter_Y = fix(Hi_res_N/2) + 1;
 
% Create coordinate system 
[x, y] = size(X3);
[X, Y] = meshgrid(1:x, 1:y);
 
% Generate low-resolution images 
for i = 1:LED_num_x * LED_num_y 
    kx = kxky_index_0(i,1);
    ky = kxky_index_0(i,2);  
    
    % Create shifted aperture mask 
    mask(:,:,i) = sqrt((X-(x/2+kx)).^2 + (Y-(y/2+ky)).^2) < kmax;
    
    % Simulate low-resolution measurement 
    X33 = fftshift(fft2(X3));
    XX = X33 .* mask(:,:,i);
    XX1 = abs(ifft2(ifftshift(XX)));
    RAW(:,:,i) = XX1; % Store low-resolution image 
end 
 
% Normalize measurements 
RAW_ori1 = RAW;
RAW = guiyi(RAW);  % Normalization function (defined elsewhere)
 
% Compute Fourier transforms of low-res images for alignment 
for j = 1:LED_num_y * LED_num_x 
    PhaseLR(:,:,j) = log(abs(fftshift(fft2(RAW(:,:,j)))) + 1); 
end 
NPhaseLR = guiyi(PhaseLR) > 0.25;  % Thresholded Fourier magnitudes 
 
%% Find centers of Fourier transforms for alignment 
centers = zeros(2, 2, LED_num_x^2);
radii = zeros(2, LED_num_x^2);
se = strel('disk', 1);
 
% Fill holes in thresholded Fourier magnitudes 
for j = 1:LED_num_y * LED_num_x 
    bw(:,:,j) = imfill(NPhaseLR(:,:,j), 'holes');
end 
 
% Find circles in Fourier transforms 
for ix = 1:LED_num_x^2 
    try 
        [c, r] = imfindcircles(NPhaseLR(:,:,ix), [40,60], 'Sensitivity', 0.95);
        centers(:,1:2,ix) = c(:,:);
        radii(:,ix) = r(1:2);
        
        % Visualize detected circles 
        imshow(NPhaseLR(:,:,ix));
        hold on;
        viscircles(centers(:,1:2,ix), radii(:,ix));
        hold off;
        pause(0.2)
        
        % Calculate approximate frequency shifts from circle centers 
        kxky_indexC0(ix,:) = (abs(centers(1,:,ix)-201) + abs(centers(2,:,ix)-201))/2;
        num_zero = find(kxky_indexC0(:,1) == 0);
    end 
end 
 
% Compare with theoretical shifts 
kxky_Check = kxky_index_0;
kxky_indexC(2:LED_num_x^2,:) = kxky_indexC0(2:LED_num_x^2,:) .* kxky_Check(2:LED_num_x^2,:) ./ abs(kxky_Check(2:LED_num_x^2,:));
 
%% Simulated annealing for misalignment correction 
theta_cor = 0;      % Corrected rotation angle 
leddx_cor = 0;      % Corrected x position 
leddy_cor = 0;      % Corrected y position 
objdx = 0;          % Objective x position (not used)
objdy = 0;          % Objective y position (not used)
 
mae_err = inf;      % Initialize error metric 
jishu_i = 0;        % Iteration counter 
xMin = zeros(1, LED_num_x * LED_num_y);
yMin = zeros(1, LED_num_x * LED_num_y);
xbest = 0; ybest = 0;  % Best position found 
tt = 2;             
de = 100;           % Initial position step size 
de1 = 1;            % Initial rotation step size 
rbest = 0;          % Best rotation found 
 
tic 
while mae_err > 0 
    % Generate random steps for position and rotation 
    xshift = de * (rand(1,3)) .* [-1, 1, 0];
    yshift = de * (rand(1,3)) .* [-1, 1, 0];
    rotation = de1 * (rand(1,3)) .* [-1, 1, 0];
    
    for Nshift = 1:3 
        % Apply step to current best estimate 
        leddx_cor = xbest + xshift(Nshift);
        leddy_cor = ybest + yshift(Nshift);
        theta_cor = rbest + rotation(Nshift);
 
        % Reduce step size after 30 iterations 
        if jishu_i > 30 
            de = de * 0.5;
            de1 = de1 * 0.5;
        end 
 
        % Calculate frequency shifts with current correction 
        for i = 1:LED_num_x 
            for j = 1:LED_num_y 
                m = [(i-(LED_num_x+1)/2), (j-(LED_num_y+1)/2)] * [cos(theta_cor*pi/180); sin(theta_cor*pi/180)];
                n = [(i-(LED_num_x+1)/2), (j-(LED_num_y+1)/2)] * [-sin(theta_cor*pi/180); cos(theta_cor*pi/180)];
                x_num = (m * LEDdelta + leddx_cor);
                y_num = (n * LEDdelta + leddy_cor);
                distance = sqrt(y_num.^2 + x_num.^2);
                theta1 = atan2(distance, LED2stage);
                kr = 1 / Lambda * sin(theta1);
                theta = atan2(y_num, x_num);
                kxky_index1(i,j,2) = (kr * sin(theta) / Pixel_size_image_freq);
                kxky_index1(i,j,1) = (kr * cos(theta) / Pixel_size_image_freq);
            end 
        end 
 
        % Reorder frequency shifts 
        kx_index = kxky_index1(:,:,1);
        ky_index = kxky_index1(:,:,2);
        kxky_indextest(:,1) = kx_index(Image_num_index(:));
        kxky_indextest(:,2) = ky_index(Image_num_index(:));
 
        % Simulate low-resolution images with current correction 
        for i = 1:LED_num_x^2 
            kx = kxky_indextest(i,1);
            ky = kxky_indextest(i,2);
            mask(:,:,i) = sqrt((X-(x/2+kx)).^2 + (Y-(y/2+ky)).^2) < kmax;
            X33 = fftshift(fft2(X3));
            XX = X33 .* mask(:,:,i);
            XX1 = abs(ifft2(ifftshift(XX)));
            RAW1(:,:,i) = guiyi(XX1);
        end 
 
        % Compute Fourier transforms of simulated images 
        for j = 1:LED_num_x^2 
            PhaseLR1(:,:,j) = log(abs(fftshift(fft2(RAW1(:,:,j)))) + 1);
        end 
        NPhaseLR1 = guiyi(PhaseLR1) > 0.25;
 
        % Calculate error between simulated and measured Fourier transforms 
        err = mean(mean(mean(abs(NPhaseLR1(200-99:200+100, 200-99:200+100, :) - ...
                               NPhaseLR(200-99:200+100, 200-99:200+100, 1:LED_num_x^2)))));
        
        % Update best estimate if error decreases 
        if err < mae_err 
            xbest = xbest + xshift(Nshift);
            ybest = ybest + yshift(Nshift);
            rbest = rbest + rotation(Nshift);
            mae_err = err;
            fprintf('Error reduced to %6.2f\n', mae_err*1e2);
            jishu_i = 0;
        end 
    end 
 
    jishu_i = jishu_i + 1;
 
    % Termination conditions 
    if mae_err <= 0.0001 || jishu_i == 50 
        xMin = xbest;
        yMin = ybest;
        rMin = rbest;
        pause(0.5);
        break 
    end 
end 
toc 
 
%% Display original and corrected misalignment parameters 
fprintf('Original parameters:\n');
fprintf('leddy = %f\n', leddy);
fprintf('leddx = %f\n', leddx);
fprintf('theta = %f\n', thetaa);
 
fprintf('\nCorrected parameters:\n');
fprintf('yMin = %f\n', yMin);
fprintf('xMin = %f\n', xMin);
fprintf('rMin = %f\n', rMin);
 
%% Reconstruct masks with corrected parameters 
for i = 1:LED_num_x 
    for j = 1:LED_num_y 
        m = [(i-(LED_num_x+1)/2), (j-(LED_num_y+1)/2)] * [cos(theta_cor*pi/180); sin(theta_cor*pi/180)];
        n = [(i-(LED_num_x+1)/2), (j-(LED_num_y+1)/2)] * [-sin(theta_cor*pi/180); cos(theta_cor*pi/180)];
        x_num = (m * LEDdelta + leddx_cor);
        y_num = (n * LEDdelta + leddy_cor);
        distance = sqrt(y_num.^2 + x_num.^2);
        theta1 = atan2(distance, LED2stage);
        kr = 1 / Lambda * sin(theta1);
        theta = atan2(y_num, x_num);
        kxky_index_cor(i,j,2) = (kr * sin(theta) / Pixel_size_image_freq);
        kxky_index_cor(i,j,1) = (kr * cos(theta) / Pixel_size_image_freq);
    end 
end 
 
% Reorder corrected frequency shifts 
kx_index_cor = kxky_index_cor(:,:,1);
ky_index_cor = kxky_index_cor(:,:,2);
kxky_index_cor0(:,1) = kx_index_cor(Image_num_index(:));
kxky_index_cor0(:,2) = ky_index_cor(Image_num_index(:));
 
% Create corrected aperture masks 
for i = 1:LED_num_x * LED_num_y 
    kx = kxky_index_cor0(i,1);
    ky = kxky_index_cor0(i,2);  
    mask(:,:,i) = sqrt((X-(x/2+kx)).^2 + (Y-(y/2+ky)).^2) < kmax;
end 
 
%% Initialize high-resolution reconstruction 
A = RAW_ori1(:,:,1);            % Start with central low-res image 
scale = Mag_image.^2;
A = A ./ scale;                 % Normalize 
[Hi_res_M, Hi_res_N] = size(A);
 
% Initialize Fourier spectrum with central low-res image 
F = fftshift(fft2(A));
Fcenter_X = fix(Hi_res_M/2) + 1;
Fcenter_Y = fix(Hi_res_N/2) + 1;
 
%% Main reconstruction loop 
Total_iter_num = 8;             % Number of reconstruction iterations 
figure(100);
 
for iter = 1:Total_iter_num 
    fprintf('Iteration %d\n', iter);
    
    %% Gerchberg-Saxton FPM iteration (PIE with alpha = 1)
    for num = 1:LED_num_x * LED_num_y 
        % Current sub-spectrum 
        Subspecturm1 = F;
        Abbr_Subspecturm1 = Subspecturm1 .* mask(:,:,num);
        
        % Low-resolution image constraint 
        Uold1 = ifft2(fftshift(Abbr_Subspecturm1));
        Unew1 = RAW_ori1(:,:,num) .* (Uold1 ./ abs(Uold1));
        
        % Calculate threshold for adaptive step size 
        Threshold(num, iter) = real(mean(mean(RAW_ori1(:,:,Image_num_index(num)) - Unew1)));
        
        % Fourier domain update 
        Abbr_Subspecturm_corrected1 = fftshift(fft2(Unew1));
        
        % Adaptive step size calculation 
        W2 = abs(mask(:,:,num)) ./ max(max(abs(mask(:,:,num))));
        invP2 = conj(mask(:,:,num)) ./ ((abs(mask(:,:,num))).^2);
        Subspecturmnew1 = Abbr_Subspecturm_corrected1 + (1.80 - W2) .* (Abbr_Subspecturm_corrected1 - Abbr_Subspecturm1) .* invP2;
        
        % Update only regions covered by current illumination 
        for a = 1:x 
            for b = 1:y 
                if mask(a,b,num) ~= 0 
                    Subspecturm1(a,b) = Subspecturmnew1(a,b);
                end 
            end 
        end 
        
        F = Subspecturm1;
        
        % Store current reconstruction 
        Result(:,:,iter) = ifft2(fftshift(F));
    end 
    
    % Calculate reconstruction error 
    if iter == 1 
        D = Result(:,:,iter) - X3;
        MSE(:,iter) = abs(real(sum(D(:).*D(:)) / numel(Result(:,:,iter))));
    else 
        D = Result(:,:,iter) - X3;
        MSE(:,iter) = abs(real(sum(D(:).*D(:)) / numel(Result(:,:,iter))));
    end 
    
    % Adaptive alpha calculation 
    if iter == 1 
        alpha = 2 * max(max(abs(Aperture_fun))) - mean(Threshold(:,iter));
    end 
    
    % Display current reconstruction 
    subplot(1,3,1)
    imshow(log(abs(F)+1), [0, max(max(log(abs(F)+1)))/2]);
    title('Fourier spectrum', 'FontSize', 20, 'FontName', 'Times New Roman');
    
    subplot(1,3,2)
    imshow(abs(Result(:,:,iter)), []);
    title('Amplitude', 'FontSize', 20, 'FontName', 'Times New Roman');
    
    subplot(1,3,3)
    imshow(imag(Result(:,:,iter)), []);
    title('Phase', 'FontSize', 20, 'FontName', 'Times New Roman');
end 
