%% Detect faces using the Viola-Jones algorithm
clc
clear all
close all
testDir = dir('../data/test_scenes/test_jpg/');

detector = vision.CascadeObjectDetector;
fopen('viola_jones_bboxes.txt','w');

fclose all;
for i = 3:size(testDir,1)
    image = imread(strcat('../data/test_scenes/test_jpg/',testDir(i).name));
    bboxes = step (detector,image);
    
    %To Show the bounding boxes adn the image:
    
    %         imshow(image)
    %         title('Face detection using Viola-Jones Algorithm')
    %         for j = 1:size(bboxes,1)
    %             rectangle('Position',bboxes(j,:),'LineWidth',5,'LineStyle','-','EdgeColor','r');
    %         end
    
    %Creating the TXT with imageid and bboxes
    fid=  fopen('viola_jones_bboxes.txt','a');
    if(~isempty(bboxes))
        for j = 1:size(bboxes,1)
            fprintf(fid,'%s %d %d %d %d \n',testDir(i).name, bboxes(j,1), bboxes(j,2),(bboxes(j,1)+bboxes(j,3)),(bboxes(j,2)+bboxes(j,4)) );
        end
    end
    fclose(fid);
%     pause
end

% Reading the TXT getting the image ids and bounding boxes.
Id = fopen('viola_jones_bboxes.txt');   
gt_info = textscan(Id, '%s %d %d %d %d');
fclose(Id);
gt_ids = gt_info{1,1};
gt_bboxes = [gt_info{1,2}, gt_info{1,3}, gt_info{1,4}, gt_info{1,5}];
gt_bboxes = double(gt_bboxes);
npos = size(gt_ids,1);
confidences = ones(npos,1);

evaluate_detections(gt_bboxes, confidences, gt_ids, '../data/test_scenes/ground_truth_bboxes_face_only.txt')
