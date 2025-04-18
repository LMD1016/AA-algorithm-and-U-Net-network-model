%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main file to implement the adaptive alpha strategy for Fourier
% ptychographic reconstruction algorithm
%
% Related Reference:
% Fast and stable Fourier ptychographic... 
% microscopy based on improved phase recovery strategy

% J. Luo, H. Tan, H. Chen,et al, submitted to Optics Express
%
% last modified on 10/31/2024
% by Luo (1552570852@qq.com)


%The code template is contributed by Chao Zuo's team. Thanks for their hard
% work and the provided original data and code---C. Zuo, J. Sun, and Q. Chen,
% "Adaptive step-size strategy for noise-robust Fourier ptychographic microscopy,"
% Optics Express 24, 20724-20744 (2016).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
clear;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize experimental parameter and HR image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eval Initialize_experimental_parameter; 
eval Initialize_image_num_index;        
eval Initialize_HR_image;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reconstruction by Gerchberg-Saxton (PIE with alpha =1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run for 8 iterations (which is sufficient to `converge')
Total_iter_num = 12;
% Reconstruction and display the result
figure
for iter = 1:Total_iter_num
    
    % Reconstruction by one iteration of Gerchberg-Saxton

    eval GS;    %Gerchberg-Saxton

    % Show the Fourier spectrum
    subplot(1,2,1)
    imshow(log(abs(F)+1),[0, max(max(log(abs(F)+1)))/2]);                                                              
    title('Fourier spectrum');
    
    % Show the reconstructed amplitude
    subplot(1,2,2)
    imshow((abs(Result)),[]);
    title(['Iteration No. = ',int2str(iter)]);
    
    pause(0.01);
end


Result_GS = Result;% Save the result

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reconstruction by the adaptive step-size strategy 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eval Initialize_HR_image;   %Re-initialize HR image

Total_iter_num = inf;

% Reconstruction and display the result
figure
for iter = 1:Total_iter_num
    
    % Reconstruction by one iteration of adaptive step-size strategy
    eval AS;
    
    % Show the Fourier spectrum
    subplot(1,2,1)
    imshow(log(abs(F)+1),[0, max(max(log(abs(F)+1)))/2]);
    title('Fourier spectrum');
    
    % Show the reconstructed amplitude
    subplot(1,2,2)
    imshow((abs(Result)),[]);
    title(['Iteration No. = ',int2str(iter), '  \alpha = ',num2str(Alpha)]);
    
    % Stop the iteration when the algorithm converges
    if(Alpha == 0) break; end
    
    pause(0.01);
end

% Save the result
Result_AS = Result;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reconstruction by the adaptive alpha strategy 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Re-initialize HR image
eval Initialize_HR_imageAA;%
Total_iter_num = 12;
% Reconstruction and display the result
figure
for iter = 1:Total_iter_num
    
    eval AA;
    
     % Show the Fourier spectrum
       subplot(1,2,1)
       imshow(log(abs(F)+1),[0, max(max(log(abs(F)+1)))/2]);
       title('Fourier spectrum');  
    % Show the reconstructed amplitude
       subplot(1,2,2)
       imshow((abs(Result(:,:,iter))),[]);
       title(['Iteration No. = ',int2str(iter)]);    
   
    pause(0.01);
end

% Save the result
Result_AA = Result(:,:,iter);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compare the  results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(9);

subplot(1,3,1)
imshow(abs(Result_GS),[]);
title('Gerchberg-Saxton');

subplot(1,3,2)
imshow(abs(Result_AS),[]);
title('Adaptive step-size');

subplot(1,3,3)
imshow(abs(Result_AA),[]);
title('Adaptive alpha');
