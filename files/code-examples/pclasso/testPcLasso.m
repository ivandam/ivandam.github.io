loadDir = '~/DATA/PCLASSO-SHARE/'; % put all your data here

%% Load image data
tmp = load([loadDir 'images.mat']);
images = tmp.images;

% load masks
tmp = load([loadDir 'labeledTemplate']);
labTemplate = tmp.labeledTemplate;
mask = labTemplate==2;

% convert pixel values to 1D vectors
% rows = ovservations, columns = variables (pixels)
dataImaging = zeros(size(images,4),nnz(mask));
for nSub = 1:size(images,4);
    img3D = images(:,:,:,nSub);
    vec = img3D(mask);
    dataImaging(nSub,:) = vec;
end

% load clinical data
dataClinical = readtable([loadDir 'clinicalData-dd.txt']);

%% Script to test the pcLassoAnalysis function
params.betaThreshold = 0.00001; % fitting coefficient threshold for including a PC term in the LASSO model
params.numCompToInclude = 5; % number of PCs to include in the LASSO regression, in order of diminishing variance explained
params.minExplained = 0.05; % threshold for including PCs into LASSO fitting based on the variance explained, e.g. 0.05 = 5%
params.pcInclusionCrit = 'explicit'; % 'explicit' or 'varexplained'; whether to use the explicit number of PCs in LASSO, or determine based on the variance explained.
params.MCReps = 500; % number of LASSO cross-validation folds
params.holdoutRatio = 0.3; % fraction of data used for testing in cross-validation
params.plotLassoPath = 1; % flag to plot LASSO MSE and fitting coefficients as functions of LASSO regularization parameter

%%
X = dataImaging;
Xref = mean(dataImaging,2); % reference variable to compare to PCLASSO model (e.g. mean pixel value within a region of interest)
subNum = [1:4 6:41];
y = dataClinical{subNum,'diseaseDuration'};
g = ones(size(y)); % grouping variable, that can be used for stratified cross-validation (e.g. when subjects come from more than one class).

output = pcLassoAnalysis(X,y,g,Xref,params);

%% Analyze the predictive performance

output.constModelFit % baseline reference - MSEs that a constant model achieves (i.e. no correlation between x and y)
output.linModelFit % performance of a linear model that uses Xref as the input
output.lassoFit % PC-LASSO predictive performance

% MSEtest: cross-validated test mean squared error
% MSEtestSE: standard error of the test MSE
% MSEall: MSE of the best cross-validated model measured on all data (without using a test set)

% betaAll: values of the fitting coefficients beta when the PCLASSO model
%           is fitted on all data.
%% Visualize the PCLASSO estimator

pcEstImg = zeros(size(mask));
pcEstImg(mask) = output.pcEstimator;

% optional smoothing
pcEstImg = imgaussfilt3(pcEstImg,0.5);
pcEstProjPos = rot90(squeeze(max(pcEstImg,[],2)));
pcEstProjNeg = rot90(squeeze(max(-pcEstImg,[],2)));

figure;
subplot(1,3,1);
imshow(pcEstProjPos,[min(pcEstProjPos(:)) max(pcEstProjPos(:))]);
title('Positive projection');

subplot(1,3,2);
imshow(pcEstProjNeg, [min(pcEstProjNeg(:)) max(pcEstProjNeg(:))]);
title('Negative projection');

subplot(1,3,3);
imshowpair(pcEstProjPos,pcEstProjNeg);
title('Positive/negative combined');

colormap(jet);

