%% Load data

load('DataD');

%% Preprocess using Z-score Normalization

[r,c] = size(fea);

%one way
feaZ = zscore(fea);

% %another way
% feaZ = zeros(r,c);
% for i = 1:c
%     meanVal = mean(fea(:,i));
%     stdVal = std(fea(:,i));
%     for j = 1:r
%         feaZ(j,i) = (fea(j,i) - meanVal) / stdVal;
%     end
% end

% the first half of data for training, the second for testing 
split = r/2;
Train_data = feaZ(1:split,:);
Train_class = gnd(1:split,:);
Test_data = feaZ(split+1:end,:);
Test_class = gnd(split+1:end,:);

%% Clssify data using K-NN classifier

k = 1:2:31;
correct_knn = zeros(1, length(k));
accuracy_knn = zeros(1, length(k));
maxAccuracy_knn = 0;

indices = crossvalind('Kfold', split, 5);

%evaluate k-NN on the values k=[1, 3, 5, 7, ..., 31]
for i = 1:length(k)
%     %one way
%     model = fitcknn(Train_data, Train_class, 'NumNeighbors', k(i), 'KFold', 5);
%     error = kfoldLoss(model);
%     accuracy = (1-error)*100;
%     if maxAccuracy_knn < accuracy
%         maxAccuracy_knn = accuracy;
%         bestk = k(i);
%     end
    
    %another way
    correct = zeros(1,5);    
    %Using 5-fold cross validation on the training set
    for j = 1:5
        crossTrain_data = Train_data((indices~=j),:);
        crossTrain_class = Train_class((indices~=j),:);
        crossTest_data = Train_data((indices==j),:);
        crossTest_class = Train_class((indices==j),:);
        
%         %knnclassify will be removed in a future release. 
%         label_knn = knnclassify(crossTest_data, crossTrain_data, crossTrain_class, k(i), 'euclidean', 'nearest');

        %Instead use fitcknn to fit a knn classification model and classify data using predict function
        model_knn = fitcknn(crossTrain_data, crossTrain_class, 'NumNeighbors', k(i));
        label_knn = predict(model_knn, crossTest_data);

        correct(j) =  sum(label_knn == crossTest_class);
    end
    correct_knn(i) = mean(correct);
    accuracy_knn(i) = correct_knn(i)/(split/5)*100;
    
    if maxAccuracy_knn < accuracy_knn(i)
        maxAccuracy_knn = accuracy_knn(i);
        bestk = k(i);
    end
end

figure;
plot(k, accuracy_knn);
xlabel('k');
ylabel('accuracy(%)');
print(gcf, 'images\K-NN', '-dpng', '-r0');

%%
%use bestk to classify all data 

model_knn_bestk = fitcknn(Train_data, Train_class, 'NumNeighbors', bestk);
label_knn_bestk = predict(model_knn_bestk, Test_data);

correct_knn_bestk =  sum(label_knn_bestk == Test_class);
accuracy_knn_bestk = correct_knn_bestk / split * 100;

%% Now that we have the best k, train and test with all data:

Accuracy_knn = zeros(1, 20);
Precision_knn = zeros(1, 20);
Recall_knn = zeros(1, 20);
Fmeasure_knn = zeros(1, 20);

trainTime = zeros(1,20);
classifyTime = zeros(1,20);

for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran= randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_class = gnd(ran(1:split),:);
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_class = gnd(ran(split+1:end),:);

    tic;
    model_knn = fitcknn(randomTrain_data, randomTrain_class, 'NumNeighbors', bestk);
    trainTime(i,j) = toc;

    tic;
    label_knn = predict(model_knn, randomTest_data);
    classifyTime(i,j) = toc;

    Correct = sum(label_knn == randomTest_class);
    tpfp = sum(label_knn == 1);
    tpfn = sum(randomTest_class == 1);
    tp = sum(label_knn == (randomTest_class == 1));

    Accuracy_knn(i) = Correct/(r-split)*100;
    Precision_knn(i) = tp/tpfp;
    Recall_knn(i) = tp/tpfn;
    Fmeasure_knn(i) = 2 / (1/Precision_knn(i) + 1/Recall_knn(i));
end

%the average and standard deviation of classification performance
avgAccuracy_knn = mean(Accuracy_knn);
stdAccuracy_knn = std(Accuracy_knn);
avgPrecision_knn = mean(Precision_knn);
stdPrecision_knn = std(Precision_knn);
avgRecall_knn = mean(Recall_knn);
stdRecall_knn = std(Recall_knn);
avgFmeasure_knn = mean(Fmeasure_knn);
stdFmeasure_knn = std(Fmeasure_knn);

trainTime_knn = mean(trainTime);
classifyTime_knn = mean(classifyTime);

%% Clssify data using RBF kernel SVM classifier
% addpath to the libsvm toolbox and data
addpath 'libsvm-3.21';
addpath 'libsvm-3.21/matlab/';
%%
c = [0.1, 0.5, 1, 2, 5, 10, 20, 50];
gamma = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10];

maxAccuracy_svm = 0;
correct_svm = zeros(length(c), length(gamma));
accuracy_svm = zeros(length(c), length(gamma));

indices = crossvalind('Kfold', split, 5);

%select soft margin penalty term "c" from the set [0.1, 0.5, 1, 2, 5, 10, 20, 50] 
for i = 1:length(c)
    %select kernel width parameter "gamma" from the set [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    for j = 1:length(gamma)               
%         %one way
%         correct = zeros(1,5);
%         %Using 5-fold cross validation on the training set
%         for k = 1:5
%             crossTrain_data = Train_data((indices~=k),:);
%             crossTrain_class = Train_class((indices~=k),:);
%             crossTest_data = Train_data((indices==k),:);
%             crossTest_class = Train_class((indices==k),:);
%             
%             SVMStruct = svmtrain(crossTrain_data, crossTrain_class, 'kernel_function', 'rbf', 'boxconstraint', c(i), 'rbf_sigma', sqrt(1/(2*gamma(j))));
%             label_svm = svmclassify(SVMStruct, crossTest_data);
%
%             correct(k) =  sum(label_svm == crossTest_class);
%         end
%         correct_svm(i, j) = mean(correct); 
%         accuracy_svm(i, j) = correct_svm(i, j) /(split/5)*100;

        %another way using libsvm fuction 
        accuracy = zeros(1,5);
        for k = 1:5
            crossTrain_data = Train_data((indices~=k),:);
            crossTrain_class = Train_class((indices~=k),:);
            crossTest_data = Train_data((indices==k),:);
            crossTest_class = Train_class((indices==k),:);
            
            %if set '-v 5', return accuracy of classification using 5-fold cross validation ranther than a svmmodel
            paramString = sprintf('-c %f -t 2 -g %f -b 1', c(i), gamma(j));
            model_svm = libsvmtrain(crossTrain_class, crossTrain_data, paramString);
            [~, accu, ~] = libsvmpredict(crossTest_class, crossTest_data, model_svm, '-b 1');
        
          accuracy(k) = accu(1);
        end
        accuracy_svm(i, j) = mean(accuracy)
        if (maxAccuracy_svm < accuracy_svm(i, j))
            maxAccuracy_svm = accuracy_svm(i, j);
            bestc = c(i);
            bestgamma = gamma(j);
        end
    end
end
%%
%use bestc and bestgamma to classify all data 

paramString = sprintf('-c %f -t 2 -g %f -b 1', bestc, bestgamma);
model_svm = libsvmtrain(Train_class, Train_data, paramString);
[label_svm, accu, prob_estimates] = libsvmpredict(Test_class, Test_data, model_svm, '-b 1');
        
%trasform Test_class
Test_target = zeros(2, r-split);
    
for j = 1:split
    if Test_class(j,1) == 1
        Test_target(1,j) = 1;
        Test_target(2,j) = 0;
    elseif Test_class(j,1) == -1
        Test_target(1,j) = 0;
        Test_target(2,j) = 1;
    end 
end

%plot ROC curve
[tpr,fpr,thresholds] = roc(Test_target, prob_estimates');
figure;
plotroc(Test_target, prob_estimates');
print(gcf, 'images\SVM-ROC', '-dpng', '-r0');

%% Now that we have the best c and gamma, train and test with all data:

Accuracy_svm = zeros(1, 20);
Precision_svm = zeros(1, 20);
Recall_svm = zeros(1, 20);
Fmeasure_svm = zeros(1, 20);

trainTime = zeros(1,20);
classifyTime = zeros(1,20);

for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran= randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_class = gnd(ran(1:split),:);
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_class = gnd(ran(split+1:end),:);
    
    paramString = sprintf('-c %f -t 2 -g %f', bestc, bestgamma);
    tic;
    model_svm = libsvmtrain(randomTrain_class, randomTrain_data, paramString);
    trainTime(i,j) = toc;

    tic;
    [label_svm, accu, prob_estimates] = libsvmpredict(randomTest_class, randomTest_data, model_svm);
    classifyTime(i,j) = toc;

    tpfp = sum(label_svm == 1);
    tpfn = sum(randomTest_class == 1);
    tp = sum(label_svm == (randomTest_class == 1));

    Accuracy_svm(i) = accu(1); 
    Precision_svm(i) = tp/tpfp;
    Recall_svm(i) = tp/tpfn;
    Fmeasure_svm(i) = 2 / (1/Precision_svm(i) + 1/Recall_svm(i));
end
%%
%the average and standard deviation of classification performance
avgAccuracy_svm = mean(Accuracy_svm);
stdAccuracy_svm = std(Accuracy_svm);
avgPrecision_svm = mean(Precision_svm);
stdPrecision_svm = std(Precision_svm);
avgRecall_svm = mean(Recall_svm);
stdRecall_svm = std(Recall_svm);
avgFmeasure_svm = mean(Fmeasure_svm);
stdFmeasure_svm = std(Fmeasure_svm);

trainTime_svm = mean(trainTime);
classifyTime_svm = mean(classifyTime);

%% Clssify data using Naive Bayes classifier

Accuary_naivebayes = zeros(1, 20);
Precision_naivebayes = zeros(1, 20);
Recall_naivebayes = zeros(1, 20);
Fmeasure_naivebayes = zeros(1, 20);

trainTime = zeros(1,20);
classifyTime = zeros(1,20);

for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran= randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_class = gnd(ran(1:split),:);
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_class = gnd(ran(split+1:end),:);
    
    tic;
    model_naivebayes = fitNaiveBayes(randomTrain_data, randomTrain_class);
    trainTime(i) = toc;
    
    tic; 
    label_naivebayes = predict(model_naivebayes, randomTest_data);
    classifyTime(i) = toc;
    
    Correct = sum(label_naivebayes == randomTest_class);
    tpfp = sum(label_naivebayes == 1);
    tpfn = sum(randomTest_class == 1);
    tp = sum(label_naivebayes == (randomTest_class == 1));

    Accuary_naivebayes(i) = Correct/(r-split)*100;
    Precision_naivebayes(i) = tp/tpfp;
    Recall_naivebayes(i) = tp/tpfn;
    Fmeasure_naivebayes(i) = 2 / (1/Precision_naivebayes(i) + 1/Recall_naivebayes(i));
end 

%the average and standard deviation of classification performance
avgAccuracy_naivebayes = mean(Accuary_naivebayes);
stdAccuracy_naivebayes = std(Accuary_naivebayes);
avgPrecision_naivebayes = mean(Precision_naivebayes);
stdPrecision_naivebayes = std(Precision_naivebayes);
avgRecall_naivebayes = mean(Recall_naivebayes);
stdRecall_naivebayes = std(Recall_naivebayes);
avgFmeasure_naivebayes = mean(Fmeasure_naivebayes);
stdFmeasure_naivebayes = std(Fmeasure_naivebayes);

trainTime_naivebayes = mean(trainTime);
classifyTime_naivebayes = mean(classifyTime);

%% Clssify data using Decision Tree classifier

Accuary_decisiontree = zeros(1,20);
Precision_decisiontree = zeros(1, 20);
Recall_decisiontree = zeros(1, 20);
Fmeasure_decisiontree = zeros(1, 20);

trainTime = zeros(1,20);
classifyTime = zeros(1,20);
for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran = randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_class = gnd(ran(1:split),:);
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_class = gnd(ran(split+1:end),:);
    
    %fitctree function returns a classification tree and fitrtree function returns a regression tree
    tic;
    model_decisiontree = fitctree(randomTrain_data, randomTrain_class);
    trainTime(i) = toc;
    
    tic;
    [label_decisiontree,score_decisiontree,node,cnum] = predict(model_decisiontree, randomTest_data);
    classifyTime(i) = toc;
    
    Correct = sum(label_decisiontree == randomTest_class);
    tpfp = sum(label_decisiontree == 1);
    tpfn = sum(randomTest_class ==1);
    tp = sum(label_decisiontree == (randomTest_class == 1));

    Accuary_decisiontree(i) = Correct/(r-split)*100;
    Precision_decisiontree(i) = tp/tpfp;
    Recall_decisiontree(i) = tp/tpfn;
    Fmeasure_decisiontree(i) = 2 / (1/Precision_decisiontree(i) + 1/Recall_decisiontree(i));
end 

%the average and standard deviation of classification performance
avgAccuracy_decisiontree = mean(Accuary_decisiontree);
stdAccuracy_decisiontree = std(Accuary_decisiontree);
avgPrecision_decisiontree = mean(Precision_decisiontree);
stdPrecision_decisiontree = std(Precision_decisiontree);
avgRecall_decisiontree = mean(Recall_decisiontree);
stdRecall_decisiontree = std(Recall_decisiontree);
avgFmeasure_decisiontree = mean(Fmeasure_decisiontree);
stdFmeasure_decisiontree = std(Fmeasure_decisiontree);

trainTime_decisiontree = mean(trainTime);
classifyTime_decisiontree = mean(classifyTime);

%% Clssify data using Neural Network classifier

Accuary_neuralnetwork = zeros(1,20);
Precision_neuralnetwork = zeros(1, 20);
Recall_neuralnetwork = zeros(1, 20);
Fmeasure_neuralnetwork = zeros(1, 20);

trainTime = zeros(1,20);
classifyTime = zeros(1,20);

for i = 1:20
    %select a random half of the data for training, the other half for testing
    ran = randperm(r);
    randomTrain_data = feaZ(ran(1:split),:);
    randomTrain_class = gnd(ran(1:split),:)';
    randomTest_data = feaZ(ran(split+1:end),:);
    randomTest_class = gnd(ran(split+1:end),:)';
    
    %trasform randomTrain_class and randomTest_class
    randomTrain_target = zeros(2,split);
    randomTest_target = zeros(2,r-split);
    
    for j = 1:split
        if randomTrain_class(1,j) == 1
            randomTrain_target(1,j) = 1;
            randomTrain_target(2,j) = 0;
        elseif randomTrain_class(1,j) == -1
            randomTrain_target(1,j) = 0;
            randomTrain_target(2,j) = 1;
        end 
    end
    
    for j = 1:split
        if randomTest_class(1,j) == 1
            randomTest_target(1,j) = 1;
            randomTest_target(2,j) = 0;
        elseif randomTest_class(1,j) == -1
            randomTest_target(1,j) = 0;
            randomTest_target(2,j) = 1;
        end 
    end
        
    %create a Neural Network
    hiddenLayerSize = 10;
    net = patternnet(hiddenLayerSize); %this is the default
    
    %train the Neural Network
    tic;
    [net, trainrecord] = train(net, randomTrain_data', randomTrain_target);
    trainTime(i) = toc;

%     nntraintool;
%     view(model_neuralnetwork);

    %test the Neural Network
    tic;
    testOutput = net(randomTest_data');
    classifyTime(i) = toc;
    
%     [cv,cm,ind,per] = confusion(randomTest_target, testOutput);
   
    label_neuralnetwork = zeros(1,r-split);
    for k = 1:r-split
        if testOutput(1, k) >= 0.5
            label_neuralnetwork(k) = 1;
        else
            label_neuralnetwork(k) = -1;
        end
    end
        
    Correct = sum(label_neuralnetwork == randomTest_class);
    tpfp = sum(label_neuralnetwork == 1);
    tpfn = sum(randomTest_class ==1);
    tp = sum(label_neuralnetwork == (randomTest_class == 1));

    Accuary_neuralnetwork(i) = Correct/(r-split)*100;
    Precision_neuralnetwork(i) = tp/tpfp;
    Recall_neuralnetwork(i) = tp/tpfn;
    Fmeasure_neuralnetwork(i) = 2 / (1/Precision_neuralnetwork(i) + 1/Recall_neuralnetwork(i));
 
%     figure;
%     plotperform(trainrecord);
%     figure;
%     plottrainstate(trainrecord);
%     figure;
%     ploterrhist(errors);
%     figure;
%     plotconfusion(randomTest_group,label_neuralnetwork);
%     figure;
%     plotroc(randomTest_group,label_neuralnetwork);
end 

%the average and standard deviation of classification performance
avgAccuracy_neuralnetwork = mean(Accuary_neuralnetwork);
stdAccuracy_neuralnetwork = std(Accuary_neuralnetwork);
avgPrecision_neuralnetwork = mean(Precision_neuralnetwork);
stdPrecision_neuralnetwork = std(Precision_neuralnetwork);
avgRecall_neuralnetwork = mean(Recall_neuralnetwork);
stdRecall_neuralnetwork = std(Recall_neuralnetwork);
avgFmeasure_neuralnetwork = mean(Fmeasure_neuralnetwork);
stdFmeasure_neuralnetwork = std(Fmeasure_neuralnetwork);

trainTime_neuralnetwork = mean(trainTime);
classifyTime_neuralnetwork = mean(classifyTime);



