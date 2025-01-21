%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize experimental parameter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load raw dataset (USAF target)
load Raw_Data;

% Raw image size
[M,N] = size(RAW(:,:,1));

% LED number
LED_num_x = 21;
LED_num_y = 21;
Total_Led = LED_num_x*LED_num_y;
LED_center = (LED_num_x*LED_num_y+1)/2;

% obj NA and magnification
NA = 0.1;
Mag = 4;

% System parameters
LED2stage = 87.5e3;
LEDdelta = 2.5e3;
Pixel_size = 6.5/Mag;
Lambda = 0.626;
k = 2*pi/Lambda;
kmax=1/Lambda*NA;

% Upsampling ratio
Mag_image = 5;
Pixel_size_image = Pixel_size/Mag_image;
Pixel_size_image_freq = 1/Pixel_size_image/(M*Mag_image);
kmax = kmax/Pixel_size_image_freq;

% Create pupil mask
[x, y] = meshgrid ...
    ((-fix(M/2):ceil(M/2)-1) ...
    ,(-fix(N/2):ceil(N/2)-1));
[Theta,R] = cart2pol(x,y);
Aperture = ~(R>kmax);
Aperture_fun = double(Aperture);
