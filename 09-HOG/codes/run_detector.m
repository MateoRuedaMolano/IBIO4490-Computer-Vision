% Starter code prepared by James Hays
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (probably 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

hog_cell_size = feature_params.hog_cell_size;
hog_cell_size = hog_cell_size(1);

% initialize these as empty and incrementally expand them
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

% differrnt scales
scales = [1, 0.95, 0.93, 0.9, 0.88, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.58, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05];
threshold = 0.5;

template_size = feature_params.template_size;


for i = 1:length(test_scenes)  
    % initialize these as empty and incrementally expand them
    c_bboxes = zeros(0,4);
    c_confidences = zeros(0,1);
    c_image_ids = cell(0,1);
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    
    % scaling
    for scale = scales
        
        % resize image
        scaled_img = imresize(img, scale);
        [height, width] = size(scaled_img);
        
        % get hog features of resized test image
        feats = vl_hog(scaled_img, hog_cell_size);
        
        % number of cells in test image
        num_cell = template_size / hog_cell_size;
        xwindow = floor(width/ hog_cell_size) - num_cell + 1;
        ywindow = floor(height / hog_cell_size) - num_cell + 1;
        
        % preallocate a matrix that store all features
        D = num_cell^2 * 31;
        feats_windows = zeros(xwindow * ywindow, D);
        
        % begin sliding window 
        for x = 1:xwindow
            for y = 1:ywindow
                windows = feats(y:(y+num_cell-1), x:(x+num_cell-1),:);
                w_index = (x-1)*ywindow + y;
                feats_windows(w_index,:) = reshape(windows, 1, D);
            end
        end
        
        % calculate the scores of all features and find the confidence is larger than threshold
        scores = feats_windows * w + b;
        index = find(scores > threshold);
        scaled_confidences = scores(index);
        
        % calculate the coordinates of bbox
        x = floor(index./ywindow);
        y = mod(index, ywindow)-1;
        xmin = (x * hog_cell_size + 1) / scale;
        ymin = (y * hog_cell_size + 1) / scale;
        xmax = (x  * hog_cell_size + template_size) / scale;
        ymax= (y * hog_cell_size + template_size) / scale;
        scaled_bboxes = [xmin, ymin,xmax, ymax];
        
        % record the image id and index of window
        scaled_image_ids = repmat({test_scenes(i).name}, size(index,1), 1);
       
        % record the current bbox coordinate and confidence
        c_bboxes      = [c_bboxes;      scaled_bboxes];
        c_confidences = [c_confidences; scaled_confidences];
        c_image_ids   = [c_image_ids;   scaled_image_ids];
    end
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [real_max] = non_max_supr_bbox(c_bboxes, c_confidences, size(img));

    c_confidences = c_confidences(real_max,:);
    c_bboxes      = c_bboxes(     real_max,:);
    c_image_ids   = c_image_ids(  real_max,:);
    
    bboxes      = [bboxes;      c_bboxes];
    confidences = [confidences; c_confidences];
    image_ids   = [image_ids;   c_image_ids];
end
end

