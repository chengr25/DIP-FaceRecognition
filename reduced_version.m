caseImg = imread('faces/s1/1.pgm');
[m,n] = size(caseImg);

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
%     img = reshape(TrainingMatrix(:,1),m,n)
%     imshow(uint8(img));
    for j = 1:3
        path = ['faces/s' num2str(i,'%d') '/' num2str(testing(j),'%d') '.pgm' ];
        img = imread(path);
        img = reshape(img,m*n, 1);
        TestingSet = [TestingSet, img];
    end
end

commonPart = mean(TrainingMatrix,2);
featurePart = [];
for i = 1:40
    tmp = TrainingMatrix(:,i) - commonPart;
    featurePart = [featurePart tmp ];
end

L = featurePart.' * featurePart;
[V,D] = eig(L);   % 40 x 40

eigenValues = diag(D);
tmpV=[];
for i = 1:40
    if(eigenValues(i)>1)
        tmpV=[tmpV V(:,i)];
    end
end
     
%V is eigen vector , D is corresponding eigen value
eigen = featurePart *  tmpV; % mn x 40

weight_vectors = [];
for i = 1:40
    tmp = eigen\featurePart(:,i);
    weight_vectors = [weight_vectors tmp];
end

count = 0;
answer = [];
for i = 1:120
    test = TestingSet(:,i);
    test = double(test);
    test = test - commonPart;
    weight = eigen\test;
    
    distance = [];
    for j = 1:40
        d = (norm(weight-weight_vectors(:,j)));
        distance = [distance d];
    end
    [mindist,index] = min(distance);
    answer = [answer index ];
    if(index == ceil(i/3))
        count = count + 1;
    end
end

accuracy = count/120;









