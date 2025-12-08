from typing import Union, Tuple
import numpy as np
import math

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