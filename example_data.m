
x = [0.1,0.1,0.1,0.3,0.3,0.3,0.5,0.5,0.5,0.7,0.7,0.7,0.9,0.9,0.9];
y = [0.8,0.5,0.6,0.1,0.4,0,0.3,0.6,0.2,0.6,0.7,0.5,0.9,1,0.6];

figure()
plot(x, y ,'rx', 'MarkerSize',12)
axis([-0.1, 1.1, -0.1, 1.1])
xlabel('temperature')
ylabel('yield')
mypdf('./../img/regress_motivation')


x = [0.125 0.625;0.375 1;0.875 0.875;1 0.625;0.75 0.125;0.625 0.625;0.625 0.375;0.375 0.375;0.375 0;0.125 0.125];
y = [0,0,0,0,0,1,1,1,1,1];

figure()
ind = find(y == 0);
plot(x(ind, 1), x(ind, 2), 'ro')
hold on
ind = find(y == 1);
plot(x(ind, 1), x(ind, 2), 'bx')
axis([-0.1, 1.1,-0.1,1.1]);
xlabel('longitude')
ylabel('latitude')
mypdf('./../img/classification_motivation')