%% Examples of benchmarks for different input formats
addpath benchmarks
clear all;close all;clc;

%% 2.   morphological version for :boundary benchmark for results stored as contour images

% imgDir = '../BSDS500/data/images/test';
% gtDir = '../BSDS500/data/groundTruth/test';
% pbDir = 'data_ourMethod1/gmm';
% outDir = 'eval_ourMethod1/test_bdry_fast';
% mkdir(outDir);
% nthresh = 3999;
% 
% tic;
% boundaryBench_fast(imgDir, gtDir, pbDir, outDir, nthresh);
% toc;


%% 4. morphological version for : all the benchmarks for results stored as a cell of segmentations

imgDir = '../BSDS500/data/images/test';
gtDir = '../BSDS500/data/groundTruth/test';
inDir = 'data_ourMethod1/gmm';
outDir = 'eval_ourMethod1/test_all_fast';
mkdir(outDir);
nthresh = 99;

tic;
allBench_fast(imgDir, gtDir, inDir, outDir, nthresh);
toc;

