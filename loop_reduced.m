caseImg = imread('faces/s1/1.pgm');
[m,n] = size(caseImg);
  % the number of feature

recorder = [];
for k = 50:100
    
    accuracy = [];
    for times = 1:20
        %load training images and testing images
        TrainingMatrix = [];
        TestingSet = [];
        TrainingSet = [];
        for i = 1:40
            training = randperm(10,7);
            testing = setdiff(randperm(10), training);
            tmp = [];
            for j = 1:7
                path = ['faces/s' num2str(i,'%d') '/' num2str(training(j),'%d') '.pgm' ];
                img = imread(path);
                img = reshape(img,m*n, 1);
                tmp = [tmp, img];
                TrainingSet = [TrainingSet img];
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
        for i = 1:280
            tmp = double(TrainingSet(:,i)) - commonPart;
            featurePart = [featurePart tmp ];
        end     

        L = featurePart.' * featurePart;
        [V,D] = eig(L);   % 280 x 280
        eigenValues = diag(D);
        [topk, indexs] = sort(eigenValues, 'descend');  
    

        tmp_eigenface = [];
        for i = 1:k
            tmp_eigenface = [tmp_eigenface V(:, indexs(i))];
        end
        eigenface = featurePart *  tmp_eigenface; % mn x k  
        projections = [];
        for i = 1:280
            tmp = eigenface' * featurePart(:,i);
            projections = [projections tmp];
        end 
        count = 0;
        answer = [];
        for i = 1:120
            test = TestingSet(:,i);
            test = double(test);
            test = test - commonPart;
            proj = eigenface' * test;
            
            distance = [];
            for j = 1:280
                d = (norm(proj-projections(:,j)));
                distance = [distance d];
            end
            [mindist,index] = min(distance);
            index = ceil(index/7);
            answer = [answer index ];
            if(index == ceil(i/3))
                count = count + 1;
            end
        end 
        tmp_rate = count/120;
        accuracy = [accuracy tmp_rate];
    end

    recorder = [recorder mean(accuracy)];
end







