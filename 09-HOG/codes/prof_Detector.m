% Starter code prepared by James Hays for CS 143, Brown University
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
%   feature_params.HOG_cell_size (default 6), the number of pixels in each
%      HOG cell. template size should be evenly divisible by HOG_cell_size.
%      Smaller HOG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality i  ncreases and more
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

% Your actual code should convert each test image to HOG feature space with
% a _single_ call to vl_HOG for each scale. Then step over the HOG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));
init_scale=1;
step_size = 1;
scale_step = 0.9; %Change
template_size = feature_params.template_size;
cell_size = feature_params(1).hog_cell_size;
cell_size = cell_size(1);

threshold = 0;



D = (feature_params.template_size / feature_params.hog_cell_size(1))^2 * 31;

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);
featureIndex= 0;




for i = 1:length(test_scenes)
    startTime = cputime;
        
    % READ THE IMAGE AND RGB2GRAY
    fprintf('Detecting faces in %s (%d/%d completed)\n', test_scenes(i).name, i, length(test_scenes));
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = im2single(img); 
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
    
        
    % FEATURES OF HOG
    a_Feat = zeros(1,D);
    faces_img = zeros(1,D);
    feat_Img = 0;
    matrix_confidences = zeros(1,1);
    matrix_bboxes = [];
    
    %SIZE OF THE IMAGE
    [a,b] = size(img);
    min_size = min(a,b);
    numRescale = 0;
    
    HOGwindow = template_size/cell_size;
    
    scale = init_scale;

    
    %Steps to rescale the image and HOG it
     while scale*min_size > template_size
        scaledImg = imresize(img,scale);
        HOG = vl_hog(scaledImg, cell_size);

        for v = 1:(size(HOG,1) - HOGwindow+1)
            for u = 1:(size(HOG,2) - HOGwindow+1)

                HOGSegment = HOG(v:v+HOGwindow-1,u:u+HOGwindow-1,:);

                HOG_size = size(HOGSegment);
                fils = HOG_size(1);
                cols = HOG_size(2);
                depth = HOG_size(3);
              

                %GET THE FEATURES OF THE SCALED IMAGE
                for j = 1:depth
                    for k = 1:fils

                        startIndex = (j-1)*(fils*cols)+(k-1)*cols + 1;
                        endIndex = (j-1)*(fils*cols)+(k-1)*cols + fils;

                        a_Feat(1, startIndex:endIndex) = double(HOGSegment(k,:,j));
                    end
                end
                %GET THE CONFIDENCE IN THIS FEATURES 
                conf = dot(w,a_Feat') + b;
                %IF conf represents a face
                if(conf > threshold)
                    %get the coordinates
                    umin = ((u-1)*cell_size / scale)+1;
                    vmin = ((v-1)*cell_size / scale)+1;
                    umax = ((u-1)+HOGwindow)*cell_size / scale;
                    vmax = ((v-1)+HOGwindow)*cell_size / scale;
                    
                    %Dimension error
                    if(umin>size(img,2) || vmin > size(img,1))
                        fprintf('scale = %d, x_min = %d, y_min = %d. But imgSize = (%d,%d)', scale,umin,vmin,size(img,2),size(img,1));
                    end
                    
                    %another feature
                    feat_Img = feat_Img+1;
                    
                    faces_img(feat_Img,:) = a_Feat;
                    matrix_confidences(feat_Img,1) = conf;
                    matrix_bboxes(feat_Img,:) = [umin, vmin, umax, vmax];
                    
                    featureIndex = featureIndex+1;
                    features(featureIndex,:) = a_Feat;
                end

            end
        end
        
        scale = scale * scale_step;
        numRescale = numRescale+ 1; %Begin with the next rescaled
    end
    matrix_img_ids(1:feat_Img,1) = {test_scenes(i).name};
    
    if feat_Img<= 0
        fprintf('%s has no faces\n',test_scenes(i).name);
        continue;
    end
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(matrix_bboxes, matrix_confidences, size(img));

    matrix_confidences = matrix_confidences(is_maximum,:);
    matrix_bboxes = matrix_bboxes(is_maximum,:);
    matrix_img_ids = matrix_img_ids(is_maximum,:);
 
    bboxes = [bboxes; matrix_bboxes];
    confidences = [confidences;matrix_confidences];
    image_ids   = [image_ids;matrix_img_ids];
    endTime = cputime;
    fprintf('Time Taken = %d\n', (endTime-startTime))
    
end
