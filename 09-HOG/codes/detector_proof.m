% Starter code prepared by James Hays for CS 143, Brown University
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [features, bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params,type)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
%     'w' - dim(1116) by 1 doubles
%     'b' - 1 x 1 double (0.574)
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

multiscale = 1; %1 for multiscale, 0 for single scale
start_scale = 1;
scale_step = 0.9;

templateSize = feature_params.template_size;
hogCellSize = feature_params.hog_cell_size;
hogCellSize = hogCellSize(1);

if strcmp(type,'hardmining')
    threshold_confidence = 0;
else
    threshold_confidence = -0.5;
end
step_size = 1;

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

featureIndex = 0;

for i = 1:length(test_scenes)
    startTime = cputime;
      
    fprintf('Detecting faces in %s (%d/%d completed)\n', test_scenes(i).name, i, length(test_scenes));
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255; %do this for pos and neg
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    
%     generating Hog Features

    tempFeature = zeros(1,D);
    facesPerImage = zeros(1,D);
    noOfFeaturesPerImage = 0;
    cur_confidences = zeros(1,1);
    cur_bboxes = [];
    
    tempImg = img;
    
    minSize = min(size(tempImg,1),size(tempImg,2));
    scale = start_scale;
    
    noOfHogPerTemplate = templateSize/hogCellSize;
    noOfRescale = 0;
    
    while scale*minSize > templateSize

        tempImg = imresize(img,scale);
        hog = vl_hog(tempImg, hogCellSize);

        for y = 1:(size(hog,1) - noOfHogPerTemplate+1)
            for x = 1:(size(hog,2) - noOfHogPerTemplate+1)

                hogSegment = hog(y:y+noOfHogPerTemplate-1,x:x+noOfHogPerTemplate-1,:);

                sizeOfHog = size(hogSegment);
                numOfMatrix = sizeOfHog(3);
                numOfRows = sizeOfHog(1);
                numOfCols = sizeOfHog(2);


                for j = 1:numOfMatrix
                    for k = 1:numOfRows

                        tempStartIndex = (j-1)*(numOfRows*numOfCols)+(k-1)*numOfCols + 1;
                        tempEndIndex = (j-1)*(numOfRows*numOfCols)+(k-1)*numOfCols + numOfRows;

                        tempFeature(1, tempStartIndex:tempEndIndex) = double(hogSegment(k,:,j));
                    end

                end
              
                tempConf = dot(w,tempFeature') + b;

                if(tempConf > threshold_confidence)

                    cur_x_min = ((x-1)*hogCellSize / scale)+1;
                    cur_y_min = ((y-1)*hogCellSize / scale)+1;
                    cur_x_max = ((x-1)+noOfHogPerTemplate)*hogCellSize / scale;
                    cur_y_max = ((y-1)+noOfHogPerTemplate)*hogCellSize / scale;
                    
                    if(cur_x_min>size(img,2) || cur_y_min > size(img,1))
                        fprintf('scale = %d, x_min = %d, y_min = %d. But imgSize = (%d,%d)', scale,cur_x_min,cur_y_min,size(img,2),size(img,1));
                    end
                    noOfFeaturesPerImage = noOfFeaturesPerImage+1;
                    facesPerImage(noOfFeaturesPerImage,:) = tempFeature;
                    cur_confidences(noOfFeaturesPerImage,1) = tempConf;
    %                 cur_bboxes(noOfFeaturesPerImage,:) = [cur_y_min, cur_x_min, (cur_y_min + templateSize-1), (cur_x_min +templateSize-1)];
                    cur_bboxes(noOfFeaturesPerImage,:) = [cur_x_min, cur_y_min, cur_x_max, cur_y_max];
                    
                    featureIndex = featureIndex+1;
                    features(featureIndex,:) = tempFeature;
                end

            end
        end
        
        if multiscale == 0 
            break;
        end
        
        scale = scale * scale_step;
        noOfRescale = noOfRescale+ 1;
    end
    cur_image_ids(1:noOfFeaturesPerImage,1) = {test_scenes(i).name};
    
    if noOfFeaturesPerImage<= 0
        fprintf('%s has no detected features!\n',test_scenes(i).name);
        continue;
    end
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, size(img));

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
    
    endTime = cputime;
    fprintf('Time Taken = %d\n', (endTime-startTime));
end
