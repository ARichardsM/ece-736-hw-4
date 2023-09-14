% Wipe data from any previous runs
clear
close all

% Control Variable
question = 3; % Question being anwsered

switch (question)
    case 1
        Question1();
    case 2
        Question2();
    case 3
        Question3();
end
%% Question 1
function Question1()
    function output = weight()
        % Set Z min and max
        zMin = 0;
        zMax = 255;

        zMid = (zMax + zMin) / 2;

        output = zeros(1,zMax);

        for i = zMin:zMax
            if (i < zMid)
                output(i+1) = i - zMin;
            else
                output(i+1) = zMax - i;
            end
        end
    end

% Control Variables
windowSer = false;
gamma = 0.1;
filenames = ["LDR_Images\1_2.jpg", "LDR_Images\1_4.jpg", "LDR_Images\1_8.jpg", "LDR_Images\1_15.jpg", "LDR_Images\1_30.jpg", "LDR_Images\1_60.jpg"];
deltaTArray = [1/2, 1/4, 1/8, 1/15, 1/30, 1/60];
lambda = 75;
weightArray = weight();

if windowSer
    filenames = ["window_series\window_exp_15_1.jpg", "window_series\window_exp_4_1.jpg", "window_series\window_exp_1_1.jpg", "window_series\window_exp_1_4.jpg", "window_series\window_exp_1_15.jpg", "window_series\window_exp_1_60.jpg", "window_series\window_exp_1_250.jpg", "window_series\window_exp_1_1000.jpg", "window_series\window_exp_1_4000.jpg"];
    deltaTArray = [15/1, 4/1, 1, 1/4, 1/15, 1/60, 1/250, 1/1000, 1/4000];
end

% Color Arrays
tmp = imread(filenames(1));
numPixels = size(tmp,1) * size(tmp,2);

redArray = zeros(numPixels, size(filenames, 2));
greenArray = zeros(numPixels, size(filenames, 2));
blueArray = zeros(numPixels, size(filenames, 2));

% Read the images and split them by colour channel
counter = 1;
for image = filenames
    loadedImage = imread(image);

    redArray(:, counter) = reshape(loadedImage(:, :, 1),1,[]);
    greenArray(:, counter) = reshape(loadedImage(:, :, 2),1,[]);
    blueArray(:, counter) = reshape(loadedImage(:, :, 3),1,[]);

    counter = counter + 1;
end
clear loadedImage counter

% Solve For g (Code from: https://cybertron.cg.tu-berlin.de/eitz/hdr/)
numExposures = size(filenames,2);

[zRed, zGreen, zBlue, sampleIndices] = makeImageMatrix(filenames, numPixels);
gRed = gsolve(zRed, log(deltaTArray), lambda, weightArray);
gGreen = gsolve(zGreen, log(deltaTArray), lambda, weightArray);
gBlue = gsolve(zBlue, log(deltaTArray), lambda, weightArray);

% Build Curves
y = (0:255);
figure
hold on
plot(gRed, y, 'r-');
xlabel('log Exposure X');
ylabel('Pixel Value Z');
hold off

figure
hold on
plot(gGreen, y, 'g-');
xlabel('log Exposure X');
ylabel('Pixel Value Z');
hold off

figure
hold on
plot(gBlue, y, 'b-');
xlabel('log Exposure X');
ylabel('Pixel Value Z');
hold off

figure
hold on
plot(gRed, y, 'r-');
plot(gGreen, y, 'g-');
plot(gBlue, y, 'b-');
xlabel('log Exposure X');
ylabel('Pixel Value Z');
hold off

% Compute Radiance Map
radianceRed = zeros(numPixels, 1);
radianceGreen = zeros(numPixels, 1);
radianceBlue = zeros(numPixels, 1);
for pixel = 1:numPixels
    numerRGB = [0, 0, 0];
    denomRGB = [0, 0, 0];

    for shutter = 1:numExposures
        numerRGB(1) = numerRGB(1) + (weightArray(redArray(pixel, shutter)+1) * (gRed(redArray(pixel, shutter)+1) - log(deltaTArray(shutter))));
        numerRGB(2) = numerRGB(2) + (weightArray(greenArray(pixel, shutter)+1) * (gGreen(greenArray(pixel, shutter)+1) - log(deltaTArray(shutter))));
        numerRGB(3) = numerRGB(3) + (weightArray(blueArray(pixel, shutter)+1) * (gBlue(blueArray(pixel, shutter)+1) - log(deltaTArray(shutter))));

        denomRGB(1) = denomRGB(1) + weightArray(redArray(pixel, shutter)+1);
        denomRGB(2) = denomRGB(2) + weightArray(greenArray(pixel, shutter)+1);
        denomRGB(3) = denomRGB(3) + weightArray(blueArray(pixel, shutter)+1);
    end

    radianceRed(pixel) = exp(numerRGB(1) / denomRGB(1));
    radianceGreen(pixel) = exp(numerRGB(2) / denomRGB(2));
    radianceBlue(pixel) = exp(numerRGB(3) / denomRGB(3));
end

% Gamma Correction
radianceRed = power(radianceRed, gamma);
radianceGreen = power(radianceGreen, gamma);
radianceBlue = power(radianceBlue, gamma);

% Tone Mapping
radianceRed = (radianceRed - min(radianceRed)) / (max(radianceRed) - min(radianceRed));
radianceGreen = (radianceGreen - min(radianceGreen)) / (max(radianceGreen) - min(radianceGreen));
radianceBlue = (radianceBlue - min(radianceBlue)) / (max(radianceBlue) - min(radianceBlue));

% Redraw Image
hdrRed = reshape(radianceRed,size(tmp,1),[]);
hdrGreen = reshape(radianceGreen,size(tmp,1),[]);
hdrBlue = reshape(radianceBlue,size(tmp,1),[]);

hdr = cat(3, hdrRed, hdrGreen, hdrBlue);

figure
imshow(hdr);
end
%% Question 2
function Question2()
% Control Variables
scale = 1;

% Data Arrays
trainErrorA = [];
testErrorA = [];
errorB = [0, 0];
basisArr = [];

% Load the Data
data = importdata('auto-mpg.data');

dataName = [];
dataVals = [];
for i = 1:size(data)
    splitData = split(data(i), '"');
    dataName = [dataName; splitData(2)];

    splitData = split(splitData(1));
    dataVals = [dataVals; transpose(splitData(1:end-1))];
end
clear splitData data

% Normalize mean and variance
dataVals = str2double(dataVals);

for i = 1:size(dataVals, 2)
    featMean = mean(dataVals(:, i));
    dataVals(:, i) = dataVals(:, i) - featMean;
end
clear featMean featSTD

% Select Training Data
testData = dataVals;
trainData = [];
for i = 1:100
    select = randi(size(testData,1));
    trainData = [trainData; testData(select,:)];
    testData(select,:) = [];
end

% Set features and targets
trainFeat = trainData(:,2:end);
trainTarg = trainData(:,1);
testFeat = testData(:,2:end);
testTarg = testData(:,1);

% For X basis functions
for numBasis = 5:5:95
    % Pick X points from the training Data
    possBasisData = trainFeat;
    basisData = [];
    for i = 1:numBasis
        select = randi(size(possBasisData,1));
        basisData = [basisData; possBasisData(select,:)];
        possBasisData(select,:) = [];
    end
    clear select possBasisData

    % Solve for Phi
    phiArr = zeros(size(trainData, 1), size(basisData, 1));

    for x = 1:size(trainData, 1)
        for phi = 1:size(basisData, 1)
            phiPoint = power(trainFeat(x, :) - basisData(phi, :), 2);
            phiPoint = exp(-phiPoint / (2 * power(scale, 2)));
            phiPoint = sum(phiPoint);

            phiArr(x, phi) = phiPoint;
        end
    end
    clear phiPoint

    % Determine weights
    weight = inv(transpose(phiArr) * phiArr) * (transpose(phiArr) * trainTarg);

    % Test weights on training set
    trainPhi = zeros(size(trainData, 1), size(basisData, 1));

    for x = 1:size(trainData, 1)
        for phi = 1:size(basisData, 1)
            phiPoint = power(trainFeat(x, :) - basisData(phi, :), 2);
            phiPoint = exp(-phiPoint / (2 * power(scale, 2)));
            phiPoint = sum(phiPoint);

            trainPhi(x, phi) = phiPoint;
        end
    end

    trainY = trainPhi * weight;

    % Determine training error
    RMS = sqrt( sum( power(trainY - trainTarg, 2) ) / size(trainY, 1));
    trainErrorA = [trainErrorA RMS];

    % Test weights on test set
    testPhi = zeros(size(testData, 1), size(basisData, 1));

    for x = 1:size(testData, 1)
        for phi = 1:size(basisData, 1)
            phiPoint = power(testFeat(x, :) - basisData(phi, :), 2);
            phiPoint = exp(-phiPoint / (2 * power(scale, 2)));
            phiPoint = sum(phiPoint);

            testPhi(x, phi) = phiPoint;
        end
    end

    testY = testPhi * weight;

    % Determine testing error
    RMS = sqrt( sum( power(testY - testTarg, 2) ) / size(testY, 1));
    testErrorA = [testErrorA RMS];

    basisArr = [basisArr numBasis];
end

figure
hold on
plot(basisArr, trainErrorA, 'g-');
plot(basisArr, testErrorA, 'r-');
legend('Training Error', 'Testing Error');
xlabel('Number of Basis Functions');
ylabel('RMS Error');
hold off

clear phi* basis* weight i x numBasis RMS testErrorA testY testPhi trainErrorA trainY trainPhi

% Select 90 Basis Functions
possBasisData = trainFeat;
basisData = [];
for i = 1:90
    select = randi(size(possBasisData,1));
    basisData = [basisData; possBasisData(select,:)];
    possBasisData(select,:) = [];
end
clear select possBasisData

% Prepare for cross validation error
crossValError = zeros(7, 1);
counter = 1;

%  For each lambda
lambdaArr = [0, 0.01, 0.1, 1, 10, 100, 1000];
for lambda = lambdaArr
    % For each fold
    for k = 1:10

        % Prepare features and targets
        remainFeat = trainFeat;
        remainTarg = trainTarg;

        crossFeat = [];
        crossTarg = [];

        for i = 0:9
            select = 10*(k) - i;
            crossTarg = [crossTarg; remainTarg(select,:)];
            remainTarg(select,:) = [];
            crossFeat = [crossFeat; remainFeat(select,:)];
            remainFeat(select,:) = [];
        end

        % Solve for Phi
        phiArr = zeros(size(crossFeat, 1), size(basisData, 1));

        for x = 1:size(crossFeat, 1)
            for phi = 1:size(basisData, 1)
                phiPoint = power(crossFeat(x, :) - basisData(phi, :), 2);
                phiPoint = exp(-phiPoint / (2 * power(scale, 2)));
                phiPoint = sum(phiPoint);

                phiArr(x, phi) = phiPoint;
            end
        end
        clear phiPoint

        % Determine weights
        weight = inv( (lambda * eye( size(phiArr, 2) )) + transpose(phiArr) * phiArr ) * (transpose(phiArr) * crossTarg);

        % Test weights on remaining set
        remainPhi = zeros(size(remainFeat, 1), size(basisData, 1));

        for x = 1:size(remainFeat, 1)
            for phi = 1:size(basisData, 1)
                phiPoint = power(remainFeat(x, :) - basisData(phi, :), 2);
                phiPoint = exp(-phiPoint / (2 * power(scale, 2)));
                phiPoint = sum(phiPoint);

                remainPhi(x, phi) = phiPoint;
            end
        end

        remainY = remainPhi * weight;

        % Determine validation error
        RMS = sqrt( sum( power(remainY - remainTarg, 2) ) / size(remainY, 1));
        crossValError(counter) = crossValError(counter) + RMS;
    end
    counter = counter + 1;
end

% Plot validation error
crossValError = crossValError / 10;

figure
semilogx(logspace(-3,3,7), crossValError);
xlabel('Lambda');
ylabel('Cross Validation Error');

% Set lambda to the best lambda
[~, I] = min(crossValError);
lambda = lambdaArr(I);
clear I

% Solve for Phi
    phiArr = zeros(size(trainData, 1), size(basisData, 1));

    for x = 1:size(trainData, 1)
        for phi = 1:size(basisData, 1)
            phiPoint = power(trainFeat(x, :) - basisData(phi, :), 2);
            phiPoint = exp(-phiPoint / (2 * power(scale, 2)));
            phiPoint = sum(phiPoint);

            phiArr(x, phi) = phiPoint;
        end
    end
    clear phiPoint

    % Determine weights
    weight = inv( (lambda * eye( size(phiArr, 2) )) + transpose(phiArr) * phiArr ) * (transpose(phiArr) * trainTarg);

% Test weights on training set
trainPhi = zeros(size(trainData, 1), size(basisData, 1));

for x = 1:size(trainData, 1)
    for phi = 1:size(basisData, 1)
        phiPoint = power(trainFeat(x, :) - basisData(phi, :), 2);
        phiPoint = exp(-phiPoint / (2 * power(scale, 2)));
        phiPoint = sum(phiPoint);

        trainPhi(x, phi) = phiPoint;
    end
end

trainY = trainPhi * weight;

% Determine training error
RMS = sqrt( sum( power(trainY - trainTarg, 2) ) / size(trainY, 1));
errorB(1) = RMS;

% Test weights on test set
testPhi = zeros(size(testData, 1), size(basisData, 1));

for x = 1:size(testData, 1)
    for phi = 1:size(basisData, 1)
        phiPoint = power(testFeat(x, :) - basisData(phi, :), 2);
        phiPoint = exp(-phiPoint / (2 * power(scale, 2)));
        phiPoint = sum(phiPoint);

        testPhi(x, phi) = phiPoint;
    end
end

testY = testPhi * weight;

% Determine testing error
RMS = sqrt( sum( power(testY - testTarg, 2) ) / size(testY, 1));
errorB(2) = RMS;

% Plot errors
figure
hold on
plot([-1], errorB(1),"ok","MarkerSize",12);
plot([1], errorB(2),"om","MarkerSize",12);
ylim([0 max(errorB)+5])
xlim([-2 2])
legend('Training Error', 'Testing Error');
xlabel('Error Type');
ylabel('RMS Error');
hold off

end
%% Question 3
function Question3()
% Pull data from the file
data = load('crabdata.txt');
output = data(:,1);
input = data(:,4:end);
clear data

% Normalize Data
for i = 1:5
    input(:,i) = ( input(:,i) - mean(input(:,i)) );
    input(:,i) = input(:,i) / max(abs(input(:,i)));
end

% Make output classes 0 or 1
output = output - 1;

% Introduce Bias
input = [ones(size(output, 1), 1), input];

% Select 25% of the data as test data
testIn = [];
trainIn = input;
testOut = [];
trainOut = output;
for i = 1:50
    select = randi(size(trainIn,1));
    testIn = [testIn; trainIn(select,:)];
    trainIn(select,:) = [];

    testOut = [testOut; trainOut(select,:)];
    trainOut(select,:) = [];
end
%clear select input output

trainIn = input(1:150, :);
trainOut = output(1:150, :);

% Calculate random initial weights
weight = rand(size(testIn, 2), 1) - 0.5;

% Calculate inital training predictions
alpha = trainIn * weight;
trainY = 1 ./ (1 + exp(-alpha));
clear alpha

% Begin interative 
for i = 1:100
    % Determine the Hessian
    R = (trainY .* (1 - trainY)) .* eye(size(trainIn, 1));
    H = transpose(trainIn) * R * trainIn;

    % Determine gradient error
    gradErr = sum( (trainY - trainOut) .* trainIn );
    gradErr = gradErr / size(trainIn, 1);

    % Adjust weights
    change = H * transpose(gradErr);
    weight = weight - H * transpose(gradErr);

    % Recalculate training predictions
    alpha = trainIn * weight;
    trainY = 1 ./ (1 + exp(-alpha));
    clear alpha
end

% Determine data for confusion matrix
trainConMat = zeros(2);
for i = 1:size(trainY, 1)
    trainConMat(trainOut(i) + 1, round(trainY(i)) + 1) = trainConMat(trainOut(i) + 1, round(trainY(i)) + 1) + 1;
end

% Print Confusion Matrix
disp("Training Confusion Matrix")
disp(trainConMat)

% Calculate the test predictions
    alpha = testIn * weight;
    testY = 1 ./ (1 + exp(-alpha));
    clear alpha

% Determine data for confusion matrix
testConMat = zeros(2);
for i = 1:size(testY, 1)
    testConMat(testOut(i) + 1, round(testY(i)) + 1) = testConMat(testOut(i) + 1, round(testY(i)) + 1) + 1;
end

% Print Confusion Matrix
disp("Testing Confusion Matrix")
disp(testConMat)

% Fit Model
svm = fitcsvm(trainIn, trainOut);

% Calculate the training predictions
trainY = predict(svm, trainIn);

% Determine data for confusion matrix
trainConMat = zeros(2);
for i = 1:size(trainY, 1)
    trainConMat(trainOut(i) + 1, round(trainY(i)) + 1) = trainConMat(trainOut(i) + 1, round(trainY(i)) + 1) + 1;
end

% Print Confusion Matrix
disp("Training Confusion Matrix")
disp(trainConMat)

% Calculate the test predictions
testY = predict(svm, testIn);

% Determine data for confusion matrix
testConMat = zeros(2);
for i = 1:size(testY, 1)
    testConMat(testOut(i) + 1, round(testY(i)) + 1) = testConMat(testOut(i) + 1, round(testY(i)) + 1) + 1;
end

% Print Confusion Matrix
disp("Testing Confusion Matrix")
disp(testConMat)

end

%% Online Scripts for Solving G

% Takes relevant samples from the images for use in gsolve.m
%
% Taken from online
function [ zRed, zGreen, zBlue, sampleIndices ] = makeImageMatrix( filenames, numPixels )
    function [ red, green, blue ] = sample( image, sampleIndices )
        % Takes relevant samples of the input image

        redChannel = image(:,:,1);
        red = redChannel(sampleIndices);

        greenChannel = image(:,:,2);
        green = greenChannel(sampleIndices);

        blueChannel = image(:,:,3);
        blue = blueChannel(sampleIndices);
    end


% determine the number of differently exposed images
numExposures = size(filenames,2);


% Create the vector of sample indices
% We need N(P-1) > (Zmax - Zmin)
% Assuming the maximum (Zmax - Zmin) = 255,
% N = (255 * 2) / (P-1) clearly fulfills this requirement
%numSamples = ceil(255*2 / (numExposures - 1)) * 2;
numSamples = ceil(255*2 / (numExposures - 1)) * 8;

% create a random sampling matrix, telling us which
% pixels of the original image we want to sample
% using ceil fits the indices into the range [1,numPixels+1],
% i.e. exactly the range of indices of zInput
step = numPixels / numSamples;
sampleIndices = floor((1:step:numPixels));
sampleIndices = sampleIndices';


% allocate resulting matrices
zRed = zeros(numSamples, numExposures);
zGreen = zeros(numSamples, numExposures);
zBlue = zeros(numSamples, numExposures);

for i=1:numExposures

    % read the nth image
    image = imread(filenames{i});

    % sample the image for each color channel
    [zRedTemp, zGreenTemp, zBlueTemp] = sample(image, sampleIndices);

    % build the resulting, small image consisting
    % of samples of the original image
    zRed(:,i) = zRedTemp;
    zGreen(:,i) = zGreenTemp;
    zBlue(:,i) = zBlueTemp;
end
end

% Code taken from Paul Debevec's SIGGRAPH'97 paper "Recovering High Dynamic Range
% Radiance Maps from Photographs"
%
%
% Given a set of pixel values observed for several pixels in several
% images with different exposure times, this function returns the
% imaging system's response function g as well as the log film irradiance
% values for the observed pixels.
%
% Assumes:
%
% Zmin = 0
% Zmax = 255
%
% Arguments:
%
% Z(i,j) is the pixel values of pixel location number i in image j
% B(j) is the log delta t, or log shutter speed, for image j
% l is lamdba, the constant that determines the amount of smoothness
% w(z) is the weighting function value for pixel value z
%
% Returns:
%
% g(z) is the log exposure corresponding to pixel value z
% lE(i) is the log film irradiance at pixel location i
%
function [g,lE]=gsolve(Z,B,l,w)
n = 256;
A = zeros(size(Z,1)*size(Z,2)+n+1,n+size(Z,1));
b = zeros(size(A,1),1);

%% Include the data-fitting equations
k = 1;
for i=1:size(Z,1)
    for j=1:size(Z,2)
        wij = w(Z(i,j)+1);
        A(k,Z(i,j)+1) = wij;
        A(k,n+i) = -wij;
        %b(k,1) = wij * B(i,j);
        b(k,1) = wij * B(j);
        k=k+1;
    end
end

%% Fix the curve by setting its middle value to 0
A(k,129) = 1;
k=k+1;

%% Include the smoothness equations
for i=1:n-2
    A(k,i)=l*w(i+1); A(k,i+1)=-2*l*w(i+1); A(k,i+2)=l*w(i+1);
    k=k+1;
end

%% Solve the system using SVD
x = A\b;
g = x(1:n);
lE = x(n+1:size(x,1));
end