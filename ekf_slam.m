function [X,Y] = cov2elli(x,P,n,NP)
% COV2ELLI Ellipse contour from Gaussian mean and covariances matrix.
% [X,Y] = COV2ELLI(X0,P) returns X and Y coordinates of the contour of
% the 1−sigma ellipse of the Gaussian defined by mean X0 and covariances
% matrix P. The contour is defined by 16 points, thus both X and Y are
% 16−vectors.
%
% [X,Y] = COV2ELLI(X0,P,n,NP) returns the n−sigma ellipse and defines the
% contour with NP points instead of the default 16 points.
%
% The ellipse can be plotted in a 2D graphic by just creating a line
% with 'line(X,Y)' or 'plot(X,Y)'.
% Copyright 2008−2009 Joan Sola @ LAAS−CNRS.
if nargin < 4
    NP = 16;
    if nargin < 3
        n = 1;
    end
end
alpha = 2*pi/NP*(0:NP); % NP angle intervals for one turn
circle = [cos(alpha);sin(alpha)]; % the unit circle
% SVD method, P = R*D*R' = R*d*d*R'
[R,D]=svd(P);
d = sqrt(D);
% n−sigma ellipse <− rotated 1−sigma ellipse <− aligned 1−sigma ellipse <− unit circle
ellip = n * R * d * circle;
% output ready for plotting (X and Y line vectors)
X = x(1)+ellip(1,:);
Y = x(2)+ellip(2,:);
end

function [pf, PF_f, PF_p] = toFrame(F , p)
% TOFRAME transform point P from global frame to frame F
%
% In:
% F : reference frame F = [f x ; f y ; f alpha]
% p : point in global frame p = [p x ; p y]22% Out:
% pf: point in frame F
% PF f: Jacobian wrt F
% PF p: Jacobian wrt p
% (c) 2010, 2011, 2012 Joan Sola
t = F(1:2);
a = F(3);
R = [cos(a) -sin(a) ; sin(a) cos(a)];
pf = R' * (p - t);
if nargout > 1 % Jacobians requested
    px = p(1);
    py = p(2);
    x  = t(1);
    y  = t(2);

    PF_f = [ ...
        -cos(a), -sin(a),  cos(a)*(py - y) - sin(a)*(px - x); ...
         sin(a), -cos(a), -cos(a)*(px - x) - sin(a)*(py - y) ];
    PF_p = R';
end
end

function [pw, PW_f, PW_pf] = fromFrame(F, pf)
% FROMFRAME Transform a point PF from local frame F to the global frame.
%
% In:
% F : reference frame F = [f x ; f y ; f alpha]
% pf: point in frame F pf = [pf x ; pf y]
% Out:
% pw: point in global frame
% PW f: Jacobian wrt F23% PW pf: Jacobian wrt pf
% (c) 2010, 2011, 2012 Joan Sola
t = F(1:2);
a = F(3);
R = [cos(a) -sin(a) ; sin(a) cos(a)];
pw = R*pf + repmat(t,1,size(pf,2)); % Allow for multiple points
if nargout > 1 % Jacobians requested
    px = pf(1);
    py = pf(2);
    PW_f = [...
        [ 1, 0, - py*cos(a) - px*sin(a)]
        [ 0, 1, px*cos(a) - py*sin(a)]];
    PW_pf = R;
end
end

function [y, Y_p] = scan (p)
% SCAN perform a range−and−bearing measure of a 2D point.
%
% In:
% p : point in sensor frame p = [p x ; p y]
% Out:
% y : measurement y = [range ; bearing]
% Y p: Jacobian wrt p24% (c) 2010, 2011, 2012 Joan Sola
px = p(1);
py = p(2);
d = sqrt(px^2+py^2);
a = atan2(py,px);
% a = atan(py/px); % use this line if you are in symbolic mode.
y = [d;a];
if nargout > 1 % Jacobians requested
    d2 = px^2 + py^2;
    Y_p = [ ...
        px/d      ,  py/d      ; ...
       -py/d2     ,  px/d2     ];
end
end

function [p, P_y] = invScan(y)
% INVSCAN Backproject a range−and−bearing measure into a 2D point.
%
% In:
% y : range−and−bearing measurement y = [range ; bearing]
% Out:
% p : point in sensor frame p = [p x ; p y]
% P y: Jacobian wrt y
% (c) 2010, 2011, 2012 Joan Sola
d = y(1);
a = y(2);
px = d*cos(a);
py = d*sin(a);
p = [px;py];
if nargout > 1 % Jacobians requested
    P_y = [...
        cos(a) , -d*sin(a)
        sin(a) , d*cos(a)];
end
end

function [ro, RO_r, RO_n] = move(r, u, n)
% MOVE Robot motion, with separated control and perturbation inputs.
%
% In:
% r: robot pose r = [x ; y ; alpha]
% u: control signal u = [d x ; d alpha]
% n: perturbation, additive to control signal
% Out:
% ro: updated robot pose
% RO r: Jacobian d(ro) / d(r)
% RO n: Jacobian d(ro) / d(n)
a = r(3);
dx = u(1) + n(1);
da = u(2) + n(2);
ao = a + da;
if ao > pi
    ao = ao - 2*pi;
end
if ao < -pi
    ao = ao + 2*pi;
end
% build position increment dp=[dx;dy], from control signal dx
dp = [dx;0];
if nargout == 1 % No Jacobians requested
    to = fromFrame(r, dp);
else % Jacobians requested26
    [to, TO_r, TO_dt] = fromFrame(r, dp);
    AO_a = 1;
    AO_da = 1;
    RO_r = [TO_r ; 0 0 AO_a];
    RO_n = [TO_dt(:,1) zeros(2,1) ; 0 AO_da];
end
ro = [to;ao];
end

function [y, Y_r, Y_p] = observe(r, p)
% OBSERVE Transform a point P to robot frame and take a
% range−and−bearing measurement.
%
% In:
% r : robot frame r = [r x ; r y ; r alpha]
% p : point in global frame p = [p x ; p y]
% Out:
% y: range−and−bearing measurement
% Y r: Jacobian wrt r
% Y p: Jacobian wrt p
% (c) 2010, 2011, 2012 Joan Sola
if nargout == 1 % No Jacobians requested
    y = scan(toFrame(r,p));
else % Jacobians requested
    [pr, PR_r, PR_p] = toFrame(r, p);
    [y, Y_pr] = scan(pr);
    % The chain rule!
    Y_r = Y_pr * PR_r;
    Y_p = Y_pr * PR_p;
end
end

function [p, P_r, P_y] = invObserve(r, y)
% INVOBSERVE Backproject a range−and−bearing measurement and transform
% to map frame.
% In:
% r : robot frame r = [r x ; r y ; r alpha]
% y : measurement y = [range ; bearing]
% Out:
% p : point in sensor frame
% P r: Jacobian wrt r
% P y: Jacobian wrt y
% (c) 2010, 2011, 2012 Joan Sola
if nargout == 1 % No Jacobians requested
    p = fromFrame(r, invScan(y));
else % Jacobians requested
    [p_r, PR_y] = invScan(y);
    [p, P_r, P_pr] = fromFrame(r, p_r);
    % here the chain rule !
    P_y = P_pr * PR_y;
end
end

% I. INITIALIZE
% I.1 SIMULATOR −− use capital letters for variable names
% W: set of external landmarks
W = cloister(-4,4,-4,4,7); % Type 'help cloister' for help
% N: number of landmarks
N = size(W,2);
% R: robot pose [x ; y ; alpha]
R = [0;-2;0];
% U: control [d x ; d alpha]
U = [0.1 ; 0.05]; % fixing advance and turn increments creates a circle
% Y: measurements of all landmarks
Y = zeros(2, N);
% I.2 ESTIMATOR
% Map: Gaussian {x,P}% x: state vector's mean
x = zeros(numel(R)+numel(W), 1);
% P: state vector's covariances matrix
P = zeros(numel(x),numel(x));
% System noise: Gaussian {0,Q}
q = [.01;.02]; % amplitude or standard deviation
Q = diag(q.^2); % covariances matrix
% Measurement noise: Gaussian {0,S}
s = [.1;1*pi/180]; % amplitude or standard deviation
S = diag(s.^2); % covariances matrix
% Map management
mapspace = false(1,numel(x)); % See Help Note #10 above.
% Landmarks management
landmarks = zeros(2, N); % See Help Note #11 above
% Place robot in map
r = find(mapspace==false, numel(R) ); % set robot pointer
mapspace(r) = true; % block map positions
x(r) = R; % initialize robot states
P(r,r) = 0; % initialize robot covariance
% I.3 GRAPHICS −− use the variable names of simulated and estimated
% variables, followed by a capital G to indicate 'graphics'.
% NOTE: the graphics code is long but absolutely necessary.
% Set figure and axes for Map
mapFig = figure(1); % create figure
cla % clear axes
axis([-6 6 -6 6]) % set axes limits
axis square % set 1:1 aspect ratio
% Simulated World −− set of all landmarks, red crosses
WG = line(...
    'linestyle','none',...
    'marker','+',...
    'color','r',...
    'xdata',W(1,:),...
    'ydata',W(2,:));
% Simulated robot, red triangle
Rshape0 = .2*[...
    2 -1 -1 2; ...
    0 1 -1 0]; % a triangle at the origin
Rshape = fromFrame(R, Rshape0); % a triangle at the robot pose
RG = line(...
    'linestyle','-',...
    'marker','none',...
    'color','r',...
    'xdata',Rshape(1,:),...
    'ydata',Rshape(2,:));
% Estimated robot, blue triangle
rG = line(...
    'linestyle','-',...
    'marker','none',...
    'color','b',...
    'xdata',Rshape(1,:),...
    'ydata',Rshape(2,:));
% Estimated robot ellipse, magenta
reG = line(...
    'linestyle','-',...
    'marker','none',...
    'color','m',...
    'xdata',[ ],...
    'ydata',[ ]);
% Estimated landmark means, blue crosses
lG = line(...
    'linestyle','none',...
    'marker','+',...
    'color','b',...
    'xdata',[ ],...
    'ydata',[ ]);
% Estimated landmark ellipses, green
leG = zeros(1,N);
for i = 1:numel(leG)
    leG(i) = line(...
        'linestyle','-',...
        'marker','none',...
        'color','g',...
        'xdata',[ ],...
        'ydata',[ ]);
end
% II. TEMPORAL LOOP
for t = 1:200
    % II.1 SIMULATOR
    % a. motion
    n = q .* randn(2,1); % perturbation vector
    R = move(R, U, zeros(2,1) ); % we will perturb the estimator
    % instead of the simulator
    % b. observations
    for i = 1:N % i: landmark index
        v = s .* randn(2,1); % measurement noise
        Y(:,i) = observe(R, W(:,i)) + v;
    end
    % II.2 ESTIMATOR
    % a. create dynamic map pointers to be used hereafter
    m = landmarks(landmarks~=0)'; % all pointers to landmarks
    rm = [r , m]; % all used states: robot and landmarks
    % ( also OK is rm = find(mapspace); )
    % b. Prediction −− robot motion
    [x(r), R_r, R_n] = move(x(r), U, n); % Estimator perturbed with n
    P(r,m) = R_r * P(r,m); % See PDF notes 'SLAM course.pdf'
    P(m,r) = P(r,m)';
    P(r,r) = R_r * P(r,r) * R_r' + R_n * Q * R_n';
    % c. Landmark correction −− known landmarks
    lids = find( landmarks(1,:) ); % returns all indices of existing landmarks
    for i = lids
        % expectation: Gaussian {e,E}
        l = landmarks(:, i)'; % landmark pointer
        [e, E_r, E_l] = observe(x(r), x(l) ); % this is h(x) in EKF
        rl = [r , l]; % pointers to robot and lmk.
        E_rl = [E_r , E_l]; % expectation Jacobian
        E = E_rl * P(rl, rl) * E_rl';
        % measurement of landmark i
        Yi = Y(:, i);
        % innovation: Gaussian {z,Z}
        z = Yi - e; % this is z = y − h(x) in EKF
        % we need values around zero for angles:
        if z(2) > pi
            z(2) = z(2) - 2*pi;
        end
        if z(2) < -pi
            z(2) = z(2) + 2*pi;
        end
        Z = S + E;
        % Individual compatibility check at Mahalanobis distance of 3−sigma
        % (See appendix of documentation file 'SLAM course.pdf')
        if z' * Z^-1 * z < 9
            % Kalman gain
            K = P(rm, rl) * E_rl' * Z^-1; % this is K = P*H'*Zˆ−1 in EKF
            % map update (use pointer rm)
            x(rm) = x(rm) + K*z;
            P(rm,rm) = P(rm,rm) - K*Z*K';
        end
    end

    % d. Landmark Initialization −− one new landmark only at each iteration
    lids = find(landmarks(1,:)==0); % all non−initialized landmarks
    if ~isempty(lids) % there are still landmarks to initialize
        i = lids(randi(numel(lids))); % pick one landmark randomly, its index is i
        l = find(mapspace==false, 2); % pointer of the new landmark in the map
        if ~isempty(l) % there is still space in the map
            mapspace(l) = true; % block map space
            landmarks(:,i) = l; % store landmark pointers
            % measurement
            Yi = Y(:,i);
            % initialization
            [x(l), L_r, L_y] = invObserve(x(r), Yi);
            P(l,rm) = L_r * P(r,rm);
            P(rm,l) = P(l,rm)';
            P(l,l) = L_r * P(r,r) * L_r' + L_y * S * L_y';
        end
    end

    % II.3 GRAPHICS
    % Simulated robot
    Rshape = fromFrame(R, Rshape0);
    set(RG, 'xdata', Rshape(1,:), 'ydata', Rshape(2,:));
    % Estimated robot
    Rshape = fromFrame(x(r), Rshape0);
    set(rG, 'xdata', Rshape(1,:), 'ydata', Rshape(2,:));
    % Estimated robot ellipse
    re = x(r(1:2)); % robot position mean
    RE = P(r(1:2),r(1:2)); % robot position covariance
    [xx,yy] = cov2elli(re,RE,3,16); % x− and y− coordinates of contour
    set(reG, 'xdata', xx, 'ydata', yy);
    % Estimated landmarks
    lids = find(landmarks(1,:)); % all indices of mapped landmarks
    lx = x(landmarks(1,lids)); % all x−coordinates
    ly = x(landmarks(2,lids)); % all y−coordinates
    set(lG, 'xdata', lx, 'ydata', ly);
    % Estimated landmark ellipses −− one per landmark
    for i = lids
        l = landmarks(:,i);
        le = x(l);
        LE = P(l,l);
        [xx,yy] = cov2elli(le,LE,3,16);
        set(leG(i), 'xdata', xx, 'ydata', yy);
    end
    % force Matlab to draw all graphic objects before next iteration
    drawnow
end



