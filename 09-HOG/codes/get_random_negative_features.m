% Starter code prepared by James Hays for CS 143, Brown University
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale, because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray
image_files = dir( fullfile( non_face_scn_path, '*.jpg' )); %Caltech no-faces stored as .jpg
num_images = length(image_files);
no_faces = [];
for i = 1:num_images
    no_faces = strcat(non_face_scn_path,'/',image_files(i).name);
%    no_faces = fullfile(image_files(i).folder,image_files(i).name);
    image = imread(no_faces); 
    images{i} = im2single(image);
end
clear i 
 %Número de ventanas por imagen

for i = 1:length(images)
    [a,b] = size(images{i});
    for j = 1:num_samples
        punto = [randi([20 a-20]),randi([20 b-20])];
        ventana{i,j} = images{i}(punto(1)-18:punto(1)+17,punto(2)-18:punto(2)+17);
    end
end

ventana = reshape(ventana,1,num_samples*num_images); %Que quede un vector
cell_size = feature_params(1).hog_cell_size;
for i = 1:length(ventana)
    HOG = vl_hog(ventana{i}, cell_size(1)); %SOLO PARA UN CELLSIZE
    features_neg(i,:) =  reshape(HOG,1,((feature_params.template_size / cell_size(1))^2 * 31));
end


% cell_size = feature_params(1).hog_cell_size;
% HOG = [];
% for i = 1:num_images
%     HOG{i} = vl_hog(im2single(images{i}), cell_size(1)); %SOLO PARA UN CELLSIZE
% end
% featured_pos = HOG;
% num_images = length(image_files);

% placeholder to be deleted - THIS ONLY WORKS FOR THE INITIAL DEMO

