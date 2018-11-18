rng(0)
x = [0.125 0.625;0.375 1;0.875 0.875;1 0.625;0.75 0.125;0.625 0.625;0.625 0.375;0.375 0.375;0.375 0;0.125 0.125];
y = [1 0; 1 0; 1 0; 1 0; 1 0; 0 1; 0 1; 0 1; 0 1; 0 1];
y_original = [0,0,0,0,0,1,1,1,1,1];

W2 = 0.5*randn(4,2); W3 = 0.5*randn(4,4); W4 = 0.5*randn(2,4);
b2 = 0.5*randn(4,1); b3 = 0.5*randn(4,1); b4 = 0.5*randn(2,1);

sigmoid = @(x, W, b) 1./(1+exp(-(W*x+b)));

for iterations = 1:5e5
    iterations
    i = randi(length(y));
    
    xi = x(i, :)';
    yi = y(i, :)';
    
    % Forward pass
    a2 = sigmoid(xi,W2,b2);
    a3 = sigmoid(a2,W3,b3);
    a4 = sigmoid(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-yi);
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % update weights
    W2 = W2 - 0.1*delta2*xi';
    W3 = W3 - 0.1*delta3*a2';
    W4 = W4 - 0.1*delta4*a3';
    b2 = b2 - 0.1*delta2;
    b3 = b3 - 0.1*delta3;
    b4 = b4 - 0.1*delta4;
end


%%

N = 100;

x1 = linspace(0,1,N);
x2 = linspace(0,1,N);
for i = 1:length(x1)
    for j = 1:length(x2) 
        xi = [x1(i); x2(j)];
        a2 = sigmoid(xi,W2,b2);
        a3 = sigmoid(a2,W3,b3);
        a4 = sigmoid(a3,W4,b4);
        y = a4;
        [m, index] = max(y);
        A(i, j) = index;
    end
end


[X,Y] = meshgrid(x1, x2);
A = A - 1;
contourf(X,Y,A)
hold on
colormap([1 1 1; 0.8 0.8 0.8])
ind = find(y_original == 0);
plot(x(ind, 1), x(ind, 2), 'ro')
ind = find(y_original == 1);
plot(x(ind, 1), x(ind, 2), 'bx')

mypdf('./../img/classification_example')