%% Load data

load('DataF');

%% Apply PCA to reduce the dimension to be 4

% %one way
% for i=1:c
%     meanval = mean(fea(:,i));
%     
%     for j=1:r
%         fea(j,i) = fea(j,i) - meanval;
%     end
% end
% covMatrix = cov(fea);
% [eigenVector, eigenValues] = eig(covMatrix);
% feaD4 = fea*eigenVector(:, end:-1:end-3);

%another way
[~,score,latent,~,~,~] = pca(fea);
feaD4 = score(:,1:4);

% Cluster the data using agglomerative algorithms in hierarchical clustering 
%% Single Link
% %one way
% tree_singlelink = linkage(feaD4, 'single');
% cluster_singlelink = cluster(tree_singlelink, 'maxclust', 10);

%another way
cluster_singlelink = clusterdata(feaD4, 'linkage', 'single', 'maxclust', 10);

% evaluate the clustering result of Single Link
%% Separation Index of Single Link:                                        

[r,c] = size(feaD4);
sumdistinc = 0;
mindist = 1000;

for i = 1:10
    feaD4_cluster = feaD4((cluster_singlelink == i), :);
    centroid = sum(feaD4_cluster,1)/sum((cluster_singlelink == i));
    sumdistinc = sumdistinc + sum((feaD4_cluster(:, 1) - centroid(1)).^2) + sum((feaD4_cluster(:, 2) - centroid(2)).^2) + sum((feaD4_cluster(:, 3)- centroid(3)).^2) + sum((feaD4_cluster(:, 4) - centroid(4)).^2);
end

for i = 1:9
    for j = i+1:10
        maxdistincs = 0;
        feaD4_cluster_i = feaD4((cluster_singlelink == i), :);
        numi = sum(cluster_singlelink == i);
        feaD4_cluster_j = feaD4((cluster_singlelink == j), :);
        numj = sum(cluster_singlelink == j);
        for k = 1:numi
            for l = 1:numj
                distincs = norm(feaD4_cluster_i(k, :) - feaD4_cluster_j(l, :));
                if maxdistincs < distincs 
                    maxdistincs = distincs;
                end
            end
        end
        if maxdistincs < mindist
            mindist = maxdistincs;
        end
    end
end

Separationindex_singlelink = sumdistinc / (r * mindist);

%% Rand index of Single Link:

[r,c] = size(feaD4);
M = (r*(r-1))/2; %total number of pairs of samples

a = 0; %number of samples in the same class and the same cluster
b = 0; %number of samples in different classes and different clusters

for i = 1:r
    for j = i+1:r
        if (gnd(i) == gnd(j)) && (cluster_singlelink(i) == cluster_singlelink(j))
            a = a + 1;
        elseif (gnd(i) ~= gnd(j)) && (cluster_singlelink(i) ~= cluster_singlelink(j))
            b = b + 1;
        end
    end
end

Randindex_singlelink = (a + b) / M;

%% F-measure of Single Link:

mij = zeros(10,10); %number of samples of Classj in Cluster i
ni = zeros(10,1); %number of samples in Clusteri
mj = zeros(10,1); %number of samples in Clusterj
precision_singlelink = zeros(10,10);
recall_singlelink = zeros(10,10);
f_singlelink = zeros(10,10);
Fmeasure_singlelink = 0;

for i = 1:r
    mij(cluster_singlelink(i), gnd(i)+1) = mij(cluster_singlelink(i), gnd(i)+1) + 1;
    ni(cluster_singlelink(i)) = ni(cluster_singlelink(i)) + 1;
    mj(gnd(i)+1) = mj(gnd(i)+1) + 1;
end

for i=1:10
    for j=1:10
        precision_singlelink(i,j) = mij(i,j) / ni(i);
        recall_singlelink(i,j) = mij(i,j) / mj(j);
        f_singlelink(i,j) = 1 / (1/precision_singlelink(i,j) + 1/recall_singlelink(i,j));
    end
end

for j=1:10
    Fmeasure_singlelink = Fmeasure_singlelink + mj(j,1) / r * max(f_singlelink(:,j));
end

%% Complete Link

cluster_completelink = clusterdata(feaD4, 'linkage', 'complete', 'maxclust', 10);

% evaluate the clustering result of Complete Link
%% Separation Index of Complete Link:                                        

[r,c] = size(feaD4);
sumdistinc = 0;
mindist = 1000;

for i = 1:10
    feaD4_cluster = feaD4((cluster_completelink == i), :);
    centroid = sum(feaD4_cluster,1)/sum((cluster_completelink == i));
    sumdistinc = sumdistinc + sum((feaD4_cluster(:, 1) - centroid(1)).^2) + sum((feaD4_cluster(:, 2) - centroid(2)).^2) + sum((feaD4_cluster(:, 3)- centroid(3)).^2) + sum((feaD4_cluster(:, 4) - centroid(4)).^2);
end

for i = 1:9
    for j = i+1:10
        maxdistincs = 0;
        feaD4_cluster_i = feaD4((cluster_completelink == i), :);
        numi = sum(cluster_completelink == i);
        feaD4_cluster_j = feaD4((cluster_completelink == j), :);
        numj = sum(cluster_completelink == j);
        for k = 1:numi
            for l = 1:numj
                distincs = norm(feaD4_cluster_i(k, :) - feaD4_cluster_j(l, :));
                if maxdistincs < distincs
                    maxdistincs = distincs;
                end
            end
        end
        if maxdistincs < mindist
            mindist = maxdistincs;
        end
    end
end

Separationindex_completelink = sumdistinc / (r * mindist);

%% Rand index of Complete Link:
[r,c] = size(feaD4);
M = (r*(r-1))/2; %total number of pairs of samples

a = 0; %number of samples in the same class and the same cluster
b = 0; %number of samples in different classes and different clusters

for i = 1:r
    for j = i+1:r
        if (gnd(i) == gnd(j)) && (cluster_completelink(i) == cluster_completelink(j))
            a = a + 1;
        elseif (gnd(i) ~= gnd(j)) && (cluster_completelink(i) ~= cluster_completelink(j))
            b = b + 1;
        end
    end
end

Randindex_completelink = (a + b) / M;

%% F-measure of Complete Link:
mij = zeros(10,10); %number of samples of Classj in Cluster i
ni = zeros(10,1); %number of samples in Clusteri
mj = zeros(10,1); %number of samples in Clusterj
precision_completelink = zeros(10,10);
recall_completelink = zeros(10,10);
f_completelink = zeros(10,10);
Fmeasure_completelink = 0;

for i = 1:r
    mij(cluster_completelink(i), gnd(i)+1) = mij(cluster_completelink(i), gnd(i)+1) + 1;
    ni(cluster_completelink(i)) = ni(cluster_completelink(i)) + 1;
    mj(gnd(i)+1) = mj(gnd(i)+1) + 1;
end

for i=1:10
    for j=1:10
        precision_completelink(i,j) = mij(i,j) / ni(i);
        recall_completelink(i,j) = mij(i,j) / mj(j);
        f_completelink(i,j) = 1 / (1/precision_completelink(i,j) + 1/recall_completelink(i,j));
    end
end

for j=1:10
    Fmeasure_completelink = Fmeasure_completelink + mj(j,1) / r * max(f_completelink(:,j));
end

%% Ward's Algorithm(minimum variance algorithm)

cluster_ward = clusterdata(feaD4, 'linkage', 'ward', 'maxclust', 10);

% evaluate the clustering result of Ward's
%% Separation Index of Ward's:                                        

[r,c] = size(feaD4);
sumdistinc = 0;
mindist = 1000;

for i = 1:10
    feaD4_cluster = feaD4((cluster_ward == i), :);
    centroid = sum(feaD4_cluster,1)/sum((cluster_ward == i));
    sumdistinc = sumdistinc + sum((feaD4_cluster(:, 1) - centroid(1)).^2) + sum((feaD4_cluster(:, 2) - centroid(2)).^2) + sum((feaD4_cluster(:, 3)- centroid(3)).^2) + sum((feaD4_cluster(:, 4) - centroid(4)).^2);
end

for i = 1:9
    for j = i+1:10
        maxdistincs = 0;
        feaD4_cluster_i = feaD4((cluster_ward == i), :);
        numi = sum(cluster_ward == i);
        feaD4_cluster_j = feaD4((cluster_ward == j), :);
        numj = sum(cluster_ward == j);
        for k = 1:numi
            for l = 1:numj
                distincs = norm(feaD4_cluster_i(k, :) - feaD4_cluster_j(l, :));
                if maxdistincs < distincs
                    maxdistincs = distincs;
                end
            end
        end
        if maxdistincs < mindist
            mindist = maxdistincs;
        end
    end
end

Separationindex_ward = sumdistinc / (r * mindist);

%% Rand index of Ward's:
[r,c] = size(feaD4);
M = (r*(r-1))/2; %total number of pairs of samples

a = 0; %number of samples in the same class and the same cluster
b = 0; %number of samples in different classes and different clusters

for i = 1:r
    for j = i+1:r
        if (gnd(i) == gnd(j)) && (cluster_ward(i) == cluster_ward(j))
            a = a + 1;
        elseif (gnd(i) ~= gnd(j)) && (cluster_ward(i) ~= cluster_ward(j))
            b = b + 1;
        end
    end
end

Randindex_ward = (a + b) / M;

%% F-measure of Ward's:
mij = zeros(10,10); %number of samples of Classj in Cluster i
ni = zeros(10,1); %number of samples in Clusteri
mj = zeros(10,1); %number of samples in Clusterj
precision_ward = zeros(10,10);
recall_ward = zeros(10,10);
f_ward = zeros(10,10);
Fmeasure_ward = 0;

for i = 1:r
    mij(cluster_ward(i), gnd(i)+1) = mij(cluster_ward(i), gnd(i)+1) + 1;
    ni(cluster_ward(i)) = ni(cluster_ward(i)) + 1;
    mj(gnd(i)+1) = mj(gnd(i)+1) + 1;
end

for i=1:10
    for j=1:10
        precision_ward(i,j) = mij(i,j) / ni(i);
        recall_ward(i,j) = mij(i,j) / mj(j);
        f_ward(i,j) = 1 / (1/precision_ward(i,j) + 1/recall_ward(i,j));
    end
end

for j=1:10
    Fmeasure_ward = Fmeasure_ward + mj(j,1) / r * max(f_ward(:,j));
end

%% Measures for all:
Part1Measures = [Separationindex_singlelink Separationindex_completelink Separationindex_ward; 
    Randindex_singlelink Randindex_completelink Randindex_ward; 
    Fmeasure_singlelink Fmeasure_completelink Fmeasure_ward];

%% the number of clusters in Ward's from 2 to 15
[r,c] = size(feaD4);
Sindex_ward = zeros(1, 14);
minSindex_ward = 1000;

for m = 2:15
    cluster_ward = clusterdata(feaD4, 'linkage', 'ward', 'maxclust', m);
    %optimal number of clusters suggested by Separation-Index   
    sumdistinc = 0;
    maxdistincs = 0;

    for i = 1:m
        feaD4_cluster = feaD4((cluster_ward == i), :);
        centroid = sum(feaD4_cluster,1)/sum((cluster_ward == i));
        sumdistinc = sumdistinc + sum((feaD4_cluster(:, 1) - centroid(1)).^2) + sum((feaD4_cluster(:, 2) - centroid(2)).^2) + sum((feaD4_cluster(:, 3)- centroid(3)).^2) + sum((feaD4_cluster(:, 4) - centroid(4)).^2);
    end

    for i = 1:m-1
        for j = i+1:m
            feaD4_cluster_i = feaD4((cluster_ward == i), :);
            numi = sum(cluster_ward == i);
            feaD4_cluster_j = feaD4((cluster_ward == j), :);
            numj = sum(cluster_ward == j);
            for k = 1:numi
                for l = 1:numj
                    distincs = norm(feaD4_cluster_i(k, :) - feaD4_cluster_j(l, :));
                    if maxdistincs < distincs
                        maxdistincs = distincs;
                    end
                end
            end
        end
    end
    Sindex_ward = sumdistinc / (r * maxdistincs);
    if  Sindex_ward < minSindex_ward
        minSindex_ward = Sindex_ward;
        bestclunum = m;
    end
end

%% Cluster the data using K-means algorithm

[r,c] = size(feaD4);
M = (r*(r-1))/2; %total number of pairs of samples

Separationindex_kmeans = zeros(1,14);
Randindex_kmeans = zeros(1,14);
Fmeasure_kmeans = zeros(1,14);

%the number of clusters from 2 to 15
for k = 2:15
    
    [cluster_kmeans,centroid] = kmeans(feaD4, k);
    
    %%
    %Separation index of K-means                                           
    sumdistinc = 0;
    mindist = 1000;

    for i = 1:k
        feaD4_cluster = feaD4((cluster_kmeans == i), :);
        centroid = sum(feaD4_cluster,1)/sum((cluster_kmeans == i));
        sumdistinc = sumdistinc + sum((feaD4_cluster(:, 1) - centroid(1)).^2) + sum((feaD4_cluster(:, 2) - centroid(2)).^2) + sum((feaD4_cluster(:, 3)- centroid(3)).^2) + sum((feaD4_cluster(:, 4) - centroid(4)).^2);
    end

    for i = 1:k-1
        for j = i+1:k
            maxdistincs = 0;
            feaD4_cluster_i = feaD4((cluster_kmeans == i), :);
            numi = sum(cluster_kmeans == i);
            feaD4_cluster_j = feaD4((cluster_kmeans == j), :);
            numj = sum(cluster_kmeans == j);
            for m = 1:numi
                for l = 1:numj
                    distincs = norm(feaD4_cluster_i(m, :) - feaD4_cluster_j(l, :));
                    if maxdistincs < distincs
                        maxdistincs = distincs;
                    end
                end
            end
            if maxdistincs < mindist
                mindist = maxdistincs;
            end
        end
    end

    Separationindex_kmeans(k-1) = sumdistinc / (r * mindist);
    
    %%
    %Rand index of K-means:
    a = 0; %number of samples in the same class and the same cluster
    b = 0; %number of samples in different classes and different clusters
    
    for i = 1:r
        for j = i+1:r
            if (gnd(i) == gnd(j)) && (cluster_kmeans(i) == cluster_kmeans(j))
                a = a + 1;
            elseif (gnd(i) ~= gnd(j)) && (cluster_kmeans(i) ~= cluster_kmeans(j))
                b = b + 1;
            end
        end
    end

    Randindex_kmeans(k-1) = (a + b) / M;

    %%
    %F-measure of kmeans:
    mij = zeros(k,10);
    ni = zeros(k,1);
    mj = zeros(10,1);
    precision = zeros(k,10);
    recall = zeros(k,10);
    f = zeros(k,10);
    
    for i = 1:r
        mij(cluster_kmeans(i), gnd(i)+1) = mij(cluster_kmeans(i), gnd(i)+1) + 1;
        ni(cluster_kmeans(i)) = ni(cluster_kmeans(i)) + 1;
        mj(gnd(i)+1) = mj(gnd(i)+1) + 1;
    end

    for i=1:k
        for j=1:10
            precision(i,j) = mij(i,j) / ni(i);
            recall(i,j) = mij(i,j) / mj(j);
            f(i,j) = 1 / (1/precision(i,j) + 1/recall(i,j));
        end
    end

    for j=1:10
        Fmeasure_kmeans(k-1) = Fmeasure_kmeans(k-1) + mj(j,1) / r * max(f(:,j));
    end
    
end

%Plot these evaluation measures with respect to the number of clusters
figure;
plot(2:15, Separationindex_kmeans, 2:15, Randindex_kmeans, 2:15, Fmeasure_kmeans);
xlabel('number of clusters');
legend({'Separation index', 'Rand index', 'F-measure'}, 'FontSize', 11); 
print(gcf, 'images\K-Means', '-dpng', '-r0');

% plot(2:15, Separationindex_kmeans);
% hold on;
% plot(2:15, Randindex_kmeans);
% hold on;
% plot(2:15, Fmeasure_kmeans);
% hold off;

%% Cluster the data using Fuzzy C-means algorithm

[r,c] = size(feaD4);

%set the exponent for partition matrix to 2
options = [2; nan; nan ; nan];
[center, membership_value, obj_fcn] = fcm(feaD4, 10, options);

avgmemvalue_digit1 = mean(membership_value(:, (gnd == 1)), 2);
avgmemvalue_digit3 = mean(membership_value(:, (gnd == 3)), 2);

figure;
plot(1:10, avgmemvalue_digit1);
xlabel('cluster 1');
print(gcf, 'images\FC-means-digit1', '-dpng', '-r0');

figure;
plot(1:10, avgmemvalue_digit3);
xlabel('cluster 3');
print(gcf, 'images\FC-means-digit3', '-dpng', '-r0');

%% hard clustering of Fuzzy C-means

%find out the clusters of samples
cluster_fcmeans = zeros(1,r);
maxmembership_value = max(membership_value);
for i = 1:r
    cluster_fcmeans(i) = find(membership_value(:,i) == maxmembership_value(i));
end

%make the max membership value of a sample to be one and the rest of its membership values zero
for i = 1:10
    for j = 1:r
        if membership_value(i,j) == maxmembership_value(j)
            membership_value(i,j) = 1;
        else
            membership_value(i,j) = 0;
        end
    end
end

% evaluate the clustering result of Fuzzy C-means
%% Separation Index of Fuzzy C-means:                                      

[r,c] = size(feaD4);
sumdistinc = 0;
mindist = 1000;

for i = 1:10
    feaD4_cluster = feaD4((cluster_fcmeans == i), :);
    centroid = sum(feaD4_cluster,1)/sum((cluster_fcmeans == i));
    sumdistinc = sumdistinc + sum((feaD4_cluster(:, 1) - centroid(1)).^2) + sum((feaD4_cluster(:, 2) - centroid(2)).^2) + sum((feaD4_cluster(:, 3)- centroid(3)).^2) + sum((feaD4_cluster(:, 4) - centroid(4)).^2);
end

for i = 1:9
    for j = i+1:10
        maxdistincs = 0;
        feaD4_cluster_i = feaD4((cluster_fcmeans == i), :);
        numi = sum(cluster_fcmeans == i);
        feaD4_cluster_j = feaD4((cluster_fcmeans == j), :);
        numj = sum(cluster_fcmeans == j);
        for m = 1:numi
            for l = 1:numj
                distincs = norm(feaD4_cluster_i(m, :) - feaD4_cluster_j(l, :));
                if maxdistincs < distincs
                    maxdistincs = distincs;
                end
            end
        end
        if maxdistincs < mindist
            mindist = maxdistincs;
        end
    end
end

Separationindex_fcmeans = sumdistinc / (r * mindist);%%%

%% Rand index of Fuzzy C-means:

[r,c] = size(feaD4);
M = (r*(r-1))/2; %total number of pairs of samples

a = 0; %number of samples in the same class and the same cluster
b = 0; %number of samples in different classes and different clusters

for i = 1:r
    for j = i+1:r
        if (gnd(i) == gnd(j)) && (cluster_fcmeans(i) == cluster_fcmeans(j))
            a = a + 1;
        elseif (gnd(i) ~= gnd(j)) && (cluster_fcmeans(i) ~= cluster_fcmeans(j))
            b = b + 1;
        end
    end
end

Randindex_fcmeans = (a + b) / M;

%% F-measure of Fuzzy C-means:

mij = zeros(10,10); %number of samples of Classj in Cluster i
ni = zeros(10,1); %number of samples in Clusteri
mj = zeros(10,1); %number of samples in Clusterj
precision_fcmeans = zeros(10,10);
recall_fcmeans = zeros(10,10);
f_fcmeans = zeros(10,10);
Fmeasure_fcmeans = 0;

for i = 1:r
    mij(cluster_fcmeans(i), gnd(i)+1) = mij(cluster_fcmeans(i), gnd(i)+1) + 1;
    ni(cluster_fcmeans(i)) = ni(cluster_fcmeans(i)) + 1;
    mj(gnd(i)+1) = mj(gnd(i)+1) + 1;
end

for i=1:10
    for j=1:10
        precision_fcmeans(i,j) = mij(i,j) / ni(i);
        recall_fcmeans(i,j) = mij(i,j) / mj(j);
        f_fcmeans(i,j) = 1 / (1/precision_fcmeans(i,j) + 1/recall_fcmeans(i,j));
    end
end

for j=1:10
    Fmeasure_fcmeans = Fmeasure_fcmeans + mj(j,1) / r * max(f_fcmeans(:,j));
end

