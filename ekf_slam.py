from typing import Union, Tuple
import numpy as np
import math
import json

class EKFSlam:
    def toFrame(self, 
                F: np.ndarray, 
                p: np.ndarray, 
                jacobians:bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Transform point p from global frame to frame F.

        Args:
            F: np.ndarray, shape (3,)
                reference frame F = [f_x, f_y, f_alpha]
            p: np.ndarray, shape (2,)
                point in global frame p = [p_x, p_y]
            jacobians: True to return Jacobians.

        Returns:
            If jacobians is False:
                pf: np.ndarray, shape (2,)
                    point in frame F
            If jacobians is True:
                (pf, PF_f, PF_p):
                    pf   : np.ndarray, shape (2,)
                    PF_f : np.ndarray, shape (2, 3), Jacobian wrt F
                    PF_p : np.ndarray, shape (2, 2), Jacobian wrt p
        """
        t = F[:2]
        a = F[2]
        R = np.array([[math.cos(a), -math.sin(a)],
                      [math.sin(a), math.cos(a)]])
        pf = R.T @ (p-t)
        if not jacobians:
            return pf
        px = p[0]
        py = p[1]
        x  = t[0]
        y  = t[1]
        PF_f = np.array([[-math.cos(a), -math.sin(a),  math.cos(a)*(py - y) - math.sin(a)*(px - x)],
                         [math.sin(a), -math.cos(a), -math.cos(a)*(px - x) - math.sin(a)*(py - y)]])
        PF_p = R.T
        return (pf, PF_f, PF_p)
    
    def fromFrame(self, 
                  F: np.ndarray, 
                  pf: np.ndarray, 
                  jacobians: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Transform a point pf from local frame F to the global frame.
        
        Args:
            F: np.ndarray, shape (3,)
                reference frame F = [f_x, f_y, f_alpha]
            pf: np.ndarray, shape (2,)
                point in frame F = [pf_x, pf_y]
            jacobians: True to return Jacobians.
            
        Returns:
            If jacobians is False:
                pw: np.ndarray, shape (2,)
                    point in global frame
            If jacobians is True:
                (pw, PW_f, PW_pf):
                    pw   : np.ndarray, shape (2,)
                    PW_f : np.ndarray, shape (2, 3), Jacobian wrt F
                    PW_pf : np.ndarray, shape (2, 2), Jacobian wrt pf
        """
        t = F[:2]
        a = F[2]
        R = np.array([[math.cos(a), -math.sin(a)],
                      [math.sin(a), math.cos(a)]])
        pw = R@pf + t
        if not jacobians:
            return pw
        px = pf[0]
        py = pf[1]
        PW_f = np.array([[ 1, 0, -py*math.cos(a) - px*math.sin(a)],
                         [ 0, 1, px*math.cos(a) - py*math.sin(a)]])
        PW_pf = R
        return (pw, PW_f, PW_pf)
    
    def scan(self, 
             p: np.ndarray, 
             jacobians: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform a range-and-bearing measure of a 2D point
        
        Args:
            p np.ndarray: point in sensor frame p = [p_x, p_y]
            jacobians: True to return Jacobians.
        
        Returns:
            If jacobians is False:
                y: np.ndarray, shape (2,)
                    measurement y = [range, bearing]
            If jacobians is True:
                (y, Y_p):
                    y   : np.ndarray, shape (2,), measurement y = [range, bearing]
                    Y_p : np.ndarray, shape (2, 3), Jacobian wrt p
        """
        px = p[0]
        py = p[1]
        d = math.hypot(px, py)
        a = math.atan2(py, px)
        y = np.array([d,a])
        if not jacobians:
            return y
        d2 = px**2 + py**2
        Y_p = np.array([[px/d, py/d],
                       [-py/d2, px/d2]])
        return (y, Y_p)

    def invScan(self, y: np.ndarray, jacobians: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Backproject a range-and-bearing measure into a 2D point.
        
        Args:
            y np.ndarray: range-and-bearing measurement y = [range ; bearing]
            jacobians: True to return Jacobians.
            
        Returns:
            If jacobians is False:
                p: np.ndarray, shape (2,)
                    point in sensor frame p = [p_x ; p_y]
            If jacobians is True:
                Tuple[p, P_y]:
                    p   : np.ndarray, shape (2,), point in sensor frame p = [p_x ; p_y]
                    P_y : np.ndarray, shape (2, 3), Jacobian wrt y
        """
        d = y[0]
        a = y[1]
        px = d*math.cos(a)
        py = d*math.sin(a)
        p = np.array([px, py])
        if not jacobians:
            return p
        P_y = np.array([[math.cos(a), -d*math.sin(a)],
                        [math.sin(a), d*math.cos(a)]])
        return (p, P_y)
    
    def move(self, 
             r: np.ndarray, 
             u: np.ndarray, 
             n: float,
             jacobians: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Robot motion, with separated control and perturbation inputs
        
        Args:
            r np.ndarray: robot pose r = [x, y, alpha]
            u np.ndarray: control signal u = [d_x, d_alpha]
            n float: perturbation, additive to control signal
            jacobians: True to return Jacobians
            
        Returns:
            If jacobians is False:
                ro: np.ndarray, shape (3,)
                    updated robot pose
            If jacobians is True:
                Tuple[ro, RO_r, RO_n]:
                    ro   : np.ndarray, shape (3,), point in sensor frame p = [p_x ; p_y]
                    RO_r : np.ndarray, shape (3, 3), Jacobian wrt r
                    RO_n : np.ndarray, shape (3, 3), Jacobian wrt n
        """
        a = r[2]
        dx = u[0] + n[0]
        da = u[1] + n[1]
        ao = a + da
        if ao > math.pi:
            ao - ao - 2*math.pi
        if ao < -math.pi:
            ao = ao + 2*math.pi
        dp = np.array([dx, 0])
        to = self.fromFrame(r, dp)
        ro = np.hstack((to,ao))
        if not jacobians:
            return ro
        to, TO_r, TO_dt = self.fromFrame(r, dp, jacobians=True)
        AO_a = 1
        AO_da = 1
        RO_r = np.vstack([TO_r, np.array([0, 0, AO_a])])
        top = np.hstack([
        TO_dt[:, [0]],           # column 0, shape (2,1)
            np.zeros((2,1))      # zeros, shape (2,1)
        ])
        bottom = np.array([[0, AO_da]])   # shape (1,2)
        RO_n = np.vstack([top, bottom])
        return (ro, RO_r, RO_n)    
    
    def observe(self,
                r: np.ndarray, 
                p: np.ndarray,
                jacobians: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Transform a point P to robot frame and take a range-and-bearing measurement.
        
        Args:
            r np.ndarray: robot frame r = [r_x, r_y, r_alpha]
            p np.ndarray: point in global frame p = [p_x, p_y]
            
        Returns:
            If jacobians is False:
                y: np.ndarray, range-and-bearing measurement
            If jacobians is True:
                Tuple[y, Y_r, Y_p]:
                    y   : np.ndarray, range-and-bearing measurement
                    Y_r : np.ndarray, Jacobian wrt r
                    Y_p : np.ndarray, Jacobian wrt p
        """
        if not jacobians:
            y = self.scan(self.toFrame(r, p))
            return y
        pr, PR_r, PR_p = self.toFrame(r, p, jacobians=True)
        y, Y_pr = self.scan(pr, jacobians=True)
        Y_r = Y_pr @ PR_r
        Y_p = Y_pr @ PR_p
        return (y, Y_r, Y_p)
    
    def invObserve(self, 
                   r: np.ndarray, 
                   y: np.ndarray,
                   jacobians: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Backproject a range-and-bearing measurement and transform to map frame.
        
        Args:
            r np.ndarray: robot frame r = [r_x, r_y, r_alpha]
            y np.ndarray: measurement y = [range, bearing]
            
        Returns:
            If jacobians is False:
                p: np.ndarray, point in sensor frame
            If jacobians is True:
                Tuple[p, P_r, P_y]:
                    p   : np.ndarray, point in sensor frame
                    P_r : np.ndarray, Jacobian wrt r
                    P_y : np.ndarray, Jacobian wrt y
        """
        if not jacobians:
            p = self.fromFrame(r, self.invScan(y))
            return p
        p_r, PR_y = self.invScan(y, jacobians=True)
        p, P_r, P_pr = self.fromFrame(r, p_r, jacobians=True)
        P_y = P_pr @ PR_y
        return (p, P_r, P_y)
    
    def cloister(self, xmin: float, xmax: float, ymin: float, ymax: float, n: int=9):
        """
        Generates features in a 2D cloister shape
        """
        x0 = (xmin+xmax)/2.0
        y0 = (ymin+ymax)/2.0
        
        hsize = xmax-xmin
        vsize = ymax-ymin
        tsize = np.diag([hsize, vsize])
        
        outer = np.arange(-(n-3)/2, (n-3)/2 + 1)
        inner = np.arange(-(n-3)/2, (n-5)/2 + 1)
        
        No = np.vstack([outer,((n-1)/2)*np.ones(outer.size)])

        Ni = np.vstack([inner,((n-3)/2)*np.ones(inner.size)])

        R = np.array([[0, -1],
                      [1,  0]])

        E = R @ np.hstack([No, Ni])

        points = np.hstack([No, Ni, E, -No, -Ni, -E])

        f = (tsize @ points) / (n - 1)

        f[0, :] += x0
        f[1, :] += y0

        return f
    
    def simulate(self):
        """
        Runs an EKF-SLAM Simulation
        """
        slam = EKFSlam()

        # Save Simulation
        self.x_hist = []
        self.P_hist = []
        self.R_hist = []

        # Init SLAM
        W = (slam.cloister(-4,4,-4,4,7)).T      # (36,2)
        N = W.shape[0]                          # int: Number of landmarks
        R = np.array([0.0, -2.0, 0.0])          # robot initial pose [x, y, alpha]
        U = np.array([0.1, 0.05])               # control vector, advance and turn increments (creates a circle)
        Y = np.zeros((N,2))                     # measurements of all landmarks
            
        # Estimator     
        # Map: Gaussian {x,P}     
        x = np.zeros(R.size + W.size)           # state vector's mean
        P = np.zeros((x.size, x.size))          # state vectors covariances matrix
            
        # System Noise: Gaussian {0, Q}     
        q = np.array([0.01, 0.02])              # amplitude or standard deviation
        Q = np.diag(q**2)                       # covariances matrix

        # Measurement Noise: Gaussian {0, S}
        s = np.array([0.1,math.pi/180])
        S = np.diag(s**2)

        # Map Management
        mapspace = np.zeros(x.size, dtype=bool)
        landmarks = np.zeros((N, 2))

        # Place Robot in Map
        r = np.where(~mapspace)[0][:R.size]     # Robot Pointer for accessing state vector
        mapspace[r] = True
        x[r] = R

        # Add Ground Truth as first state in History
        self.x_hist.append(np.hstack((R, W.flatten())))
        self.P_hist.append(P.copy())
        self.R_hist.append(R.copy())

        # Temporal Loop
        for t in range(0, 200):
            # Motion    
            n = q * np.random.randn(2)         # perturbation vector
            R = slam.move(R, U, np.zeros((2))) # Move the robot
            for i in range(0,N):               # i landmark index
                v = s * np.random.randn(2)     # measurement noise
                Y[i, :] = slam.observe(R, W[i, :]) + v
            m = landmarks[landmarks != 0].T.astype(int)
            rm = np.hstack([r, m]).astype(int)
            # Predict
            x[r], R_r, R_n = slam.move(x[r], U, n, jacobians=True)

            if m.size > 0:
                P[np.ix_(r, m)] = R_r @ P[np.ix_(r, m)]
                P[np.ix_(m,r)] = P[np.ix_(r,m)].T
            P[np.ix_(r, r)] = (R_r @ P[np.ix_(r, r)] @ R_r.T + R_n @ Q @ R_n.T)
            
            lids = np.where(landmarks[0, :] != 0)[0]
            for i in lids:
                # expectation: Gaussian {e,E}
                l = landmarks[i, :].T.astype(int)                      # landmark pointer
                e, E_r, E_l = slam.observe(x[r], x[l], jacobians=True) # this is h(x) in EKF
                rl = np.hstack([r , l]).astype(int)                    # pointers to robot and lmk.
                E_rl = np.hstack([E_r , E_l])                          # expectation Jacobian
                E = E_rl @ P[np.ix_(rl, rl)] @ E_rl.T
                # measurement of landmark i
                Yi = Y[i, :]
                # innovation: Gaussian {z,Z}
                z = Yi - e                                             # this is z = y − h(x) in EKF
                # we need values around zero for angles:
                if z[1] > math.pi:
                    z[1] = z[1] - 2*math.pi
                if z[1] < -math.pi:
                    z[1] = z[1] + 2*math.pi
                Z = S + E
                # Individual compatibility check at Mahalanobis distance of 3−sigma
                # (See appendix of documentation file 'SLAM course.pdf')
                Z_inv = np.linalg.inv(Z)
                d2 = float(z.T @ Z_inv @ z)
                if d2 < 9:
                    # Kalman gain
                    K = P[np.ix_(rm, rl)] @ E_rl.T @ Z_inv             # this is K = P*H'*Zˆ−1 in EKF
                    # map update (use pointer rm)
                    x[rm] = x[rm] + K @ z
                    P[np.ix_(rm,rm)] = P[np.ix_(rm,rm)] - K @ Z @ K.T
                    
            # Landmark Initialization −− one new landmark only at each iteration
            lids = np.where(landmarks[:, 0] == 0)[0]                   # all non−initialized landmarks
            if len(lids) > 0:                                          # there are still landmarks to initialize
                i = np.random.choice(lids)                             # pick one landmark randomly, its index is i
                l = np.flatnonzero(~mapspace)[:2]                      # pointer of the new landmark in the map
                if len(l) > 0:                                         # there is still space in the map
                    mapspace[l] = True                                 # block map space
                    landmarks[i,:] = l                                 # store landmark pointers
                    # measurement
                    Yi = Y[i,:]
                    # initialization
                    x[l], L_r, L_y = slam.invObserve(x[r], Yi, jacobians=True)
                    P[np.ix_(l,rm)] = L_r @ P[np.ix_(r,rm)]
                    P[np.ix_(rm,l)] = P[np.ix_(l,rm)].T
                    P[np.ix_(l,l)] = L_r @ P[np.ix_(r,r)] @ L_r.T + L_y @ S @ L_y.T
                    
            self.x_hist.append(x.copy())
            self.P_hist.append(P.copy())
            self.R_hist.append(R.copy())
            
            l_init = x[3:].reshape(36,2)
            l_init = l_init[~np.all(l_init == 0, axis=1)]
            
        final_x = self.x_hist[-1]      # (75,)
        final_P = self.P_hist[-1]      # (75,75)

        distances = []
        euclid = []

        for i in range(N):
            l = landmarks[i, :].astype(int)

            # true and estimated positions (2D)
            true_pos = W[i, :]
            est_pos  = final_x[l]

            # covariance block for landmark
            P_ll = final_P[np.ix_(l, l)]

            # Euclidean error
            diff = true_pos - est_pos
            euclid.append(float(np.linalg.norm(diff)))

            # Mahalanobis distance
            invP_ll = np.linalg.inv(P_ll)
            d2 = diff.T @ invP_ll @ diff
            d = d2**0.5

            distances.append(d)
            
        distances = np.array(distances)
        euclid = np.array(distances)
            
        d2 = distances**2

        print(f"mean Mahalanobis: {distances.mean():.2f}")
        print(f"mean Euclidean  : {euclid.mean():.2f}")
        print(f"mean d^2        : {d2.mean():.2f}")
            
    def save(self, output_file="slam_history.jsonl", decimals=4):

        with open(output_file, "w") as f:
            for i in range(len(self.x_hist)):
                x_form = self.x_hist[i].squeeze()[3:].reshape((36,2))
                x_form = x_form[~np.all(x_form == 0, axis=1)] # Ignore landmarks not yet initialized at time stamp i
                
                entry = {
                    "robot_position": np.round(self.R_hist[i], decimals).tolist(),
                    "map": np.round(x_form, decimals).tolist()
                }
                f.write(json.dumps(entry) + "\n")