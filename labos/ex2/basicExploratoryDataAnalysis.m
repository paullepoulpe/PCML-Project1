clear
clc
close all

%% Exercise 1
%% 1)
A = eye(3);
%% 2)
A = diag([1 2 3]);
%% 3)
A = zeros(3);
%% 4)
M = rand(4,3);
%% 5)
T = reshape(M,[6,2]);
%% 6)
a = [1; 2; 3];

%% Exercise 2
%% 1)
b = M*a;
%% 2)
lengthB = length(b);
%% 3)
e = b'*b;

%% Exercise 3
%% 1)
A = rand(3);
%% 2)
c = A*a;
%% 3)
d = A^(-1)*c;
%% 4)
lengthD = length(d);
%% 5)
% Inverse operation

%% Exercise 5
%% 1)
A = [1 2; 3 4];
%% 2)
k = 3.1;
%% 3) 
B = A + k - k;
%% 4) 
C = A - B;

%% Exercise 6
c = 5;
A = rand(3);
B = rand(3);
C = A*B;

%% Exercise 7
%% 1)
x = -10:0.1:10;
y = 100*((cos(x)).^2)+x.^3;
figure()
plot(x,y);
%% 2)
figure();
plot(x,y,'r');
%% 3) 
figure();
plot(x,y,'.');

%% Exercise 8
x = rand(10000,1);
figure()
histogram(x);

%% Exercise 9
x = randn(10000,1).*2 + 10;
std(x)
mean(x)
figure()
histogram(x);

%% Exercise 10
clear
load('height_weight_gender.mat')
%
height = height.*0.025;
weight = weight.*0.454;
males_nb = sum(gender); % 5000
females_nb = length(gender)-males_nb;
%
maleMeanHeight = 0;
maleMeanWeight= 0;
femaleMeanHeight = 0;
femaleMeanWeight= 0;
jW=1;
jH=1;
kW=1;
kH=1;
for i=1:length(gender)
    if gender(i)==1
        maleMeanHeight = maleMeanHeight + height(i);
        maleMeanWeight = maleMeanWeight + weight(i);
        maleHeight(jH) = height(i);
        jH = jH+1;
        maleWeight(jW) = weight(i);
        jW = jW+1;
    else
        femaleMeanHeight = femaleMeanHeight + height(i);
        femaleMeanWeight = femaleMeanWeight + weight(i);
        femaleHeight(kH) = height(i);
        kH = kH+1;
        femaleWeight(kW) = weight(i);
        kW = kW+1;
    end
end

maleMeanHeight = maleMeanHeight/males_nb;
maleMeanWeight= maleMeanWeight/males_nb;
femaleMeanHeight = femaleMeanHeight/females_nb;
femaleMeanWeight= femaleMeanWeight/females_nb;

%
meanWeightFemaleGivenSize = 0;
numberFemaleGivenSize = 0;

for i=1:length(gender)
    if gender(i)==0
        if height(i)>1.6
            if height(i)<1.7
                meanWeightFemaleGivenSize = meanWeightFemaleGivenSize + weight(i);
                numberFemaleGivenSize = numberFemaleGivenSize + 1;
            end
        end
    end
end

meanWeightFemaleGivenSize = meanWeightFemaleGivenSize/numberFemaleGivenSize;

%% Exercise 11
figure()
ax(1) = subplot(321);
histogram(weight);
title('pop weight');
ax(3) = subplot(323);
histogram(maleWeight);
title('male weight');
ax(5) = subplot(325);
histogram(femaleWeight);
title('female weight');
ax(2) = subplot(322);
histogram(height);
title('pop height');
ax(4) = subplot(324);
histogram(maleHeight);
title('male height');
ax(6) = subplot(326);
histogram(femaleHeight);
title('female height');

%% Exercise 12
figure()
plot(height,weight,'.');
xlabel('height');
ylabel('weight');
















