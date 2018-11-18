x = [0.1,0.1,0.1,0.3,0.3,0.3,0.5,0.5,0.5,0.7,0.7,0.7,0.9,0.9,0.9];
y = [0.8,0.5,0.6,0.1,0.4,0,0.3,0.6,0.2,0.6,0.7,0.5,0.9,1,0.6];

W2 = 0.5*randn(4,1); W3 = 0.5*randn(4,4); W4 = 0.5*randn(1,4);
b2 = 0.5*randn(4,1); b3 = 0.5*randn(4,1); b4 = 0.5*randn(1,1);

sigmoid = @(x, W, b) 1./(1+exp(-(W*x+b)));

for iterations = 1:5e5
    iterations
    i = randi(length(y));
    
    % Forward pass
    a2 = sigmoid(x(i),W2,b2);
    a3 = sigmoid(a2,W3,b3);
    a4 = sigmoid(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-y(i));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % update weights
    W2 = W2 - 0.1*delta2*x(i)';
    W3 = W3 - 0.1*delta3*a2';
    W4 = W4 - 0.1*delta4*a3';
    b2 = b2 - 0.1*delta2;
    b3 = b3 - 0.1*delta3;
    b4 = b4 - 0.1*delta4;
end



figure
plot(x, y ,'rx', 'MarkerSize',12)
hold on
xticks([-1,0,1])
yticks([0,1])

x = [0:0.01:1];
for i= 1:length(x)
    a2 = sigmoid(x(i),W2,b2);
    a3 = sigmoid(a2,W3,b3);
    a4 = sigmoid(a3,W4,b4);
    y(i) = a4;
end
plot(x,y)
mypdf('./../img/regression_example')