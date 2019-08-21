function output = pcLassoAnalysis(X,y,g,Xref,params)
% Perform PC-LASSO analysis

% X: rows are observations, columns are variables
% y: outcome variable

if isfield(params,'betaThreshold')
    betaThreshold = params.betaThreshold;
else
    betaThreshold = 0.00001; % fitting coefficient threshold for including term in the model
end

if isfield(params,'numCompToInclude')
    numCompToInclude = params.numCompToInclude;
else
    numCompToInclude = inf; % fitting coefficient threshold for including term in the model
end

if isfield(params,'minExplained')
    minExplained = params.minExplained;
else
    minExplained = 0.05; % fitting coefficient threshold for including term in the model
end

if isfield(params,'pcInclusionCrit')
    pcInclusionCrit = params.pcInclusionCrit;
else
    pcInclusionCrit = 'explicit'; % fitting coefficient threshold for including term in the model
end

if isfield(params,'MCReps')
    MCReps = params.MCReps;
else
    MCReps = 500; % number of LASSO cross-validation folds
end

if isfield(params,'holdoutRatio')
    holdoutRatio = params.holdoutRatio;
else
    holdoutRatio = 0.3; % fraction of data used for testing in cross-validation
end

if isfield(params,'plotLassoPath')
    plotLassoPath = params.plotLassoPath;
else
    plotLassoPath = 0; % fraction of data used for testing in cross-validation
end

numObs = size(X,1); % number of observations
numVariables = size(X,2); % number of variables

%% PCA - compute loadings and corresponding scores

[pCoeff,pScores,latent,tsquared,explained,mu] = pca(X);
numCompTotal = size(pCoeff,2);

% determine how many components to include in LASSO regression
numAboveThreshold = nnz(explained >= minExplained*100);

if strcmp(pcInclusionCrit,'varexplained')
    numCompToInclude = numAboveThreshold;
else
    if numCompToInclude==-1
        numCompToInclude = numCompTotal;
    end
end

%% Compute PC-LASSO MSE
Xpc = pScores(:,1:numCompToInclude);

Y = y;
G = g;

xNames = cell(1,numCompToInclude);
for nC = 1:numCompToInclude
    xNames{1,nC} = ['PC' num2str(nC)];
end

rng(0); % for reproducibility
[Ball, FitInfo, Btest] = lassoAdvanced(Xpc,Y,G,xNames,MCReps,holdoutRatio,'CVPlots',plotLassoPath);
bestLambdaIndex = FitInfo.IndexMinMSE; % index of best lambda step

lassoFit.MSEtest = FitInfo.MSE(bestLambdaIndex); % mean test MSE
lassoFit.MSEtestSE = FitInfo.MSEse(bestLambdaIndex); % standard error of mean test MSE
lassoFit.MSEall = FitInfo.MSEall(bestLambdaIndex); % all MSE
lassoFit.selectedVars = find(abs(Ball(:,bestLambdaIndex)')>betaThreshold); % included terms
lassoFit.betaAll = Ball(:,bestLambdaIndex);
%figure; histogram(FitInfo.MSEtrials(:,FitInfo.IndexMinMSE));

%% MSE with Linear and constant models

Y = y;

% linear model
rng(0); % for reproducibility
[MSEtest,MSEtestse, Ypred, matMSE] = linModelMse(Xref,Y,holdoutRatio,MCReps);
linModelFit.MSEtest = MSEtest;
linModelFit.MSEtestSE = MSEtestse;
linModelFit.MSEall = mean((Ypred-Y).^2);

% constant model
rng(0); % for reproducibility
[MSEtest,MSEtestse, Ypred] = constModelMse(Xref,Y,holdoutRatio,MCReps);
constModelFit.MSEtest = MSEtest;
constModelFit.MSEtestSE = MSEtestse;
constModelFit.MSEall = mean((Ypred-Y).^2);

%% Compute PC estimator

betas = Ball(:,bestLambdaIndex);
betasNorm = betas/norm(betas);

pcEstimator = zeros(1,numVariables);
for nComp = 1:numCompToInclude
    compWeights = pCoeff(:,nComp)';
    pcEstimator = pcEstimator + compWeights*betasNorm(nComp);
    %ha.TitleFontSizeMultiplier = 0.5;
end

%% Compile output

output.lassoFit = lassoFit;
output.linModelFit = linModelFit;
output.constModelFit = constModelFit;
output.pcEstimator = pcEstimator;

end

%% Functions

function [meanMSE, seMSE, yPred, matMSE] = linModelMse(X,y,holdoutRatio,MCReps)
% perform linear fitting and estimate MSE
numberOfSamples = size(X,1);
numTest = round(numberOfSamples*(holdoutRatio));
numTrain = numberOfSamples - numTest;

matMSE = zeros(MCReps,1);
for nRep = 1:MCReps
    indOrder = randperm(numberOfSamples);
    indTrain = indOrder(1:numTrain);
    indTest = indOrder(numTrain+1:end);
    xTrain = X(indTrain,:);
    yTrain = y(indTrain);
    xTest = X(indTest,:);
    yTest = y(indTest);
    
    mdl = fitlm(xTrain,yTrain);
    
    [yPred] = predict(mdl,xTest);
    matMSE(nRep,1) = mean((yPred-yTest).^2);
end
meanMSE = mean(matMSE);
seMSE = std(matMSE)/sqrt(MCReps);
%figure;
%histogram(matMSE);
mdl = fitlm(X,y);
[yPred] = predict(mdl,X);
% plot prediction with full data
end

function [meanMSE, seMSE, yPred] = constModelMse(x,y,holdoutRatio,MCReps)
% estimate MSE of the model given by the mean of the data
numberOfSamples = size(x,1);
numTest = round(numberOfSamples*(holdoutRatio));
numTrain = numberOfSamples - numTest;

matMSE = zeros(MCReps,1);
for nRep = 1:MCReps
    indOrder = randperm(numberOfSamples);
    indTrain = indOrder(1:numTrain);
    indTest = indOrder(numTrain+1:end);
    xTrain = x(indTrain);
    yTrain = y(indTrain);
    xTest = x(indTest);
    yTest = y(indTest);
    
    yTrainMean = mean(yTrain);
    
    matMSE(nRep,1) = mean((yTest-yTrainMean).^2);
end
meanMSE = mean(matMSE);
seMSE = std(matMSE)/sqrt(MCReps);
yMean = mean(y);
yPred = yMean;
end

function [B, FitInfo, Bcv] = lassoAdvanced(X,Y,G,PredictorNames,MCReps,HoldoutRatio,varargin)
% Function input:
%
% X - Numeric matrix with n rows and p columns. Each row represents on
% observation, and each column represents one predictor(variable).
% Y - Numeric vector of length n, where n is the number of rows of X. Y(i)
% is the response to row i of X.
% G - Numeric vector of length n, where n is the number of rows of X. G(i)
% is the group label of row i. During cross-validation, observations with
% unique labels are groupped and the same proportion of samples for
% training/testing is drawn from each group (stratified cross-validation)
%
% PredictorNames - Cell array of size 1 x p containing strings, naming each
% input (optional).  If not provided, predictor variables are named
% 'x1','x2',... in the order of columns in X.
%
% MCReps - Positive integer, the number of Monte Carlo repetitions for
% cross-validation (holdout cross-validation is always used)
% HoldoutRatio - scalar from 0-1, indicating percentage of data used for
% testing (e.g. percentHoldout=0.3 will use 60% of the samples for
% training and 30% for testing).
%
% (Optional Arguments)
% CVPlots - Supply 1 to produce plots during cross-validation, default 0 (off):
%   1. MSE Trace with SE Bars vs Lambda
%   2. Coefficient Trace vs. Lambda

% default function parameters
plot_flag = 0;

% Optional arguments
optargin = size(varargin,2);
i = 1;
while i <= optargin
    switch lower(varargin{i})
        case 'cvplots'
            plot_flag = varargin{i+1};
    end
    i = i+2;
end

% create stratified partition matrix
nobs = size(X,1);
nvars = size(X,2);
ttmat = zeros(nobs,MCReps);
for ntt = 1:MCReps
    rng(ntt);
    cvp = cvpartition(G,'HoldOut',HoldoutRatio);
    if ntt == 1
        % take first cvp for MATLAB LASSO use
        cvp_matlab = cvp;
    end
    ttmat(:,ntt) = cvp.training;
end
ttmat = logical(ttmat);

% manual lasso
opt = statset('UseParallel',true); % Parallel for lasso
% determine sequence of lambdas
[~, STATS0] = lasso(X,Y,'CV',cvp_matlab,'MCReps',MCReps,'Standardize',0,'Options',opt);
lda.seq = STATS0.Lambda;
nlam = length(lda.seq);

% initialize cross-validation outputs
output.XvalErr = zeros(MCReps,nlam);
output.coef = zeros(nvars,nlam,MCReps);

% cross-validation for lambda fine-tuning
for nfold = 1:MCReps
    % choose fold
    traindx = ttmat(:,nfold);
    testdx = ~traindx;

    xxtrain = X(traindx,:);
    xxtest = X(testdx,:);
    yytrain = Y(traindx,:);
    yytest = Y(testdx,:);

    % lasso
    [B_ft, STATS_ft] = lasso(xxtrain,yytrain,'Lambda',lda.seq,'CV','resubstitution','Standardize',0,'Options',opt);
    intcpt = repmat(STATS_ft.Intercept,size(xxtest,1),1);

    % predict on test set for all lambdas
    yhatmat = xxtest*B_ft + intcpt;
    % find true cross-validation error for all lambdas
    msemat = mean((yhatmat-repmat(yytest,1,100)).^2);

    % each model outputs
    output.XvalErr(nfold,:) = msemat;
    output.coef(:,:,nfold) = B_ft;

end % nfold

% cross-validation outputs
output.meanXvalErr = mean(output.XvalErr,1);
output.stdXvalErr = std(output.XvalErr,1,1);
output.minMse = min(output.meanXvalErr,[],2);
output.minIndex = find(output.meanXvalErr == output.minMse);
output.optimlam = lda.seq(output.minIndex);

% model selection after lambda fine-tuning
[B, FitInfo_MATLAB] = lasso(X,Y,'Lambda',lda.seq,'CV','resubstitution','Standardize',0,'Options',opt);

% predicted Y values
Ypred = zeros(numel(Y),numel(lda.seq));
for nSeq = 1:numel(lda.seq)
    for nTerm = 1:size(X,2)
        Ypred(:,nSeq) = Ypred(:,nSeq) + B(nTerm,nSeq)*X(:,nTerm);
    end
    Ypred(:,nSeq) = Ypred(:,nSeq) + FitInfo_MATLAB.Intercept(nSeq);
end

% training coefficients during cross-validation
Bcv = output.coef;

% Information of model fit
FitInfo.Intercept = FitInfo_MATLAB.Intercept;
FitInfo.Lambda = lda.seq;
FitInfo.MSE = output.meanXvalErr;
FitInfo.MSEtrials = output.XvalErr;
FitInfo.MSEall = FitInfo_MATLAB.MSE; % MSE on all data without cross-validation
FitInfo.Ypred = Ypred;
FitInfo.MSEse = output.stdXvalErr/sqrt(MCReps);
FitInfo.MSEstd = output.stdXvalErr;
FitInfo.LambdaMinMSE = output.optimlam;
FitInfo.IndexMinMSE = output.minIndex;

if plot_flag == 1
    fig = figure;
    subplot(1,2,1);
    % Plot 1: MSE Trace with SE Error Bars
    SEval = FitInfo.MSEse;
    P1 = errorbar(lda.seq,output.meanXvalErr,SEval,'o','MarkerSize',5,'MarkerFaceColor','black'); hold on
    P1.Color = [0.5 0.5 0.5];
    % plot lambda min. mse
    yminmax = ylim;
    lambdaLine = plot([output.optimlam output.optimlam],yminmax,'g--');
    xlabel('Regularization Parameter \Lambda');
    ylabel('MSE with SE Errorbars');
    title('Cross-Validated MSE of LASSO Fit');
    legend(lambdaLine,'\Lambda_{CV Min. MSE}');      
    set(gca,'XScale','log');
     % Plot 2: Coefficient Trace Plot
    subplot(1,2,2); 
    hold on
    for ncf = 1:size(B,1);
        trace_coef = B(ncf,:);
        plot(lda.seq,trace_coef);
    end
    % plot lambda min mse
    yminmax = fig.CurrentAxes.YLim;
    plot([output.optimlam output.optimlam],yminmax,'g--');
    xlabel('Regularization Parameter \Lambda');
    ylabel('Coefficient Trace');
    title('Trace Plot of Coefficients Fit by LASSO')
    legend([PredictorNames '\Lambda_{CV Min. MSE}']);
    set(gca,'XScale','log');
end

end


