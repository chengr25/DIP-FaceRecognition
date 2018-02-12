caseImg = imread('faces/s1/1.pgm');
[m,n] = size(caseImg);
k= 51;  % the number of feature

%load training images and testing images
TrainingMatrix = [];
TestingSet = [];
for i = 1:40
    training = randperm(10,7);
    testing = setdiff(randperm(10), training);
    tmp = [];
    for j = 1:7
        path = ['faces/s' num2str(i,'%d') '/' num2str(training(j),'%d') '.pgm' ];
        img = imread(path);
        img = reshape(img,m*n, 1);
        tmp = [tmp, img];
    end
    TrainingMatrix = [TrainingMatrix, mean(tmp,2)];
    for j = 1:3
        path = ['faces/s' num2str(i,'%d') '/' num2str(testing(j),'%d') '.pgm' ];
        img = imread(path);
        img = reshape(img,m*n, 1);
        TestingSet = [TestingSet, img];
    end
end

%training images minus average vector to get the unique Part of each image
commonPart = mean(TrainingMatrix,2);
featurePart = [];
for i = 1:40
    tmp = TrainingMatrix(:,i) - commonPart;
    featurePart = [featurePart tmp ];
end

%compute covariance matrix, and get eigenvector matrix and eigenvalue
C = (featurePart * featurePart.')/39; %mn x mn
[V,D] = eig(C);    %V is eigen vector , D is corresponding eigen value
eigenValues = diag(D);
eigenface = [];
[a,b] = size(V);
[topk, indexs ] = sort(eigenValues,'descend');

%pick k feature vector as eigenface
for i = 1:k
    eigenface = [eigenface V(:, indexs(i))];
end

%calculate projection of each training image
projections = []; 
for i = 1:40
    tmp = eigenface' * featurePart(:,i);
    projections = [projections tmp];
end

%classify the testing image
count = 0;
answer = [];
for i = 1:120
    test = double(TestingSet(:,i));
    test = test - commonPart;
    projection = eigenface' * test;
    distance = [];
    for j = 1:40
        d = (norm(projection-projections(:,j)));
        distance = [distance d];
    end
    [mindist,index] = min(distance);
    answer = [answer index];
    if(index == ceil(i/3))
        count = count + 1;
    end
end          



