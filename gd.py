import numpy as np
class gd:
    '''
    Includes 8 variations of gradient descent
    
    pv: Plain Vanilla
    
    momentum: Momentum 
    
    adam: Adam
    
    adamax: AdaMax
    
    nadam: Nesterov-accelerated Adaptive Moment Estimation
    
    amsgrad: AMSGrad
    
    nag: Nesterov accelerated gradient
    
    RMSprop: RMSprop
    
    Each function returns the corresponding x, y and z path taken during the descent.
    
    
    Example:
    -------
    
    test = gd(fn, fn_grad)
    test.pv() or test.momentum() or test.adam()
    
    '''
    
    def __init__(self, fn, fn_grad):
        '''
        fn should include two variables x and y
        Example:
        -------
        fn = lambda x,y: x*y + y
        
        
        fn_grad should call the corresponding fn and x and y
        Note that autograd only takes float values of x and y
        
        Example:
        -------

        def fn_grad(fn, x1, x2):
            dy_dx1 = elementwise_grad(fn, argnum=0)(x1, x2)
            dy_dx2 = elementwise_grad(fn, argnum=1)(x1, x2)

            return dy_dx1, dy_dx2
        
        
        '''
        self.fn = fn
        self.fn_grad = fn_grad
        
    def pv(self, x_init, y_init, n_iter, lr, tol= 1e-5):
        '''
        Plain vanilla gradient descent that attempts to find the local minima by descending down each gradient.
        Note the learning rate, small learning rates means reaching the minima may take longer while a large
        learning rate could cause us to miss the minima and bounce around.

        Parameters
        ----------

        x_init: Start point at coordinate x as float. MUST BE FLOAT (e.g 1.0)

        y_init: Start point at coordinate y as float. MUST BE FLOAT (e.g 1.0)

        n_iter: Number of iteration as interger

        lr: Learning Rate given as float, size of learning rate affects each correction/step taken in descent

        tol: Tolerance rate to end descent, by default should be set at a zero equivalent


        Return
        ------

        Path of x, y and z taken to map the route of gradient descent. Returns as 3 one dimensional arrays. 


        '''


        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)
            x += -lr * dx
            y += -lr * dy
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  Plain Vanilla  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  Plain Vanilla  \033[0m \nDid not converge')
        else:
            print('\033[1m  Plain Vanilla  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_pv = z_path
        self.z_pv = z
        return x_path,y_path,z_path

    
    def momentum(self, x_init, y_init, n_iter, lr, beta, tol= 1e-5):
        '''
        Accelerates the plain vanilla gradient descent. This inertia allows it to overcome certain local
        minima and pass small humps. This acceleration is achieved by giving gradient descent a short
        term memory element.

        Parameters
        ----------

        x_init: Start point at coordinate x as float. MUST BE FLOAT (e.g 1.0)

        y_init: Start point at coordinate y as float. MUST BE FLOAT (e.g 1.0)

        n_iter: Number of iteration as interger

        lr: Learning Rate given as float, size of learning rate affects each correction/step taken in descent

        beta: The memory/acceleration element that affects momentum. Beta=0 gives us plain vanilla gradient descent

        tol: Tolerance rate to end descent, by default should be set at a zero equivalent


        Return
        ------

        Path of x, y and z taken to map the route of gradient descent. Returns as 3 one dimensional arrays. 
        
        
        Reference
        ---------
        
        Source: https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
        
        '''

        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        mu_x = 0
        mu_y = 0

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)

            mu_x = beta * mu_x + lr * dx
            mu_y = beta * mu_y + lr * dy

            x += -mu_x
            y += -mu_y
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  Momentum  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  Momentum  \033[0m \nDid not converge')
        else:
            print('\033[1m  Momentum  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_momentum = z_path
        self.z_momentum = z
        return x_path,y_path,z_path
    
    
    def adam(self, x_init, y_init, n_iter, lr, beta_1 = .9, beta_2 = .99, tol= 1e-5, epsilon = 1e-8):
        '''
        Adaptive Moment Estimation (Adam) computes adaptive learning rates for each parameter. Adam 
        stores exponentially decaying average of past squared gradients (v) and keeps an 
        exponentially decaying average of past gradients (m)

        Parameters
        ----------

        x_init: Start point at coordinate x as float. MUST BE FLOAT (e.g 1.0)

        y_init: Start point at coordinate y as float. MUST BE FLOAT (e.g 1.0)

        n_iter: Number of iteration as interger

        lr: Learning Rate given as float, size of learning rate affects each correction/step taken in descent

        beta_1: Float. Decay rate for the first moment estimates (mean)(m). Default = 0.9

        beta_2: Float. Decay rate for the second moment estimates (uncentered variance)(v). Default = 0.99

        tol: Tolerance rate to end descent, by Default = zero equivalent

        epsilon: Float constant. Used for numerical stability. Default = 10**-8


        Return
        ------

        Path of x, y and z taken to map the route of gradient descent. Returns as 3 one dimensional arrays. 
        
        Reference
        ---------
        
        Source: http://ruder.io/optimizing-gradient-descent/index.html#adam
                https://arxiv.org/abs/1412.6980

        '''

        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        m_x = 0
        v_x = 0
        m_y = 0
        v_y = 0

        lr_t = lr * (np.sqrt(1 - beta_2))/(1 - beta_1)

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)

            m_x = beta_1 * m_x + dx * (1 - beta_1)
            v_x = beta_2 * v_x + dx * dx * (1 - beta_2)
            m_x_hat = m_x / (1 - np.power(beta_1, max(i, 1)))
            v_x_hat = v_x / (1 - np.power(beta_2, max(i, 1)))

            m_y = beta_1 * m_y + dy * (1 - beta_1)
            v_y = beta_2 * v_y + dy * dy * (1 - beta_2)
            m_y_hat = m_y / (1 - np.power(beta_1, max(i, 1)))
            v_y_hat = v_y / (1 - np.power(beta_2, max(i, 1)))

            x += - (lr_t * m_x_hat)/(np.sqrt(v_x_hat) + epsilon)
            y += - (lr_t * m_y_hat)/(np.sqrt(v_y_hat) + epsilon)
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  Adam  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  Adam  \033[0m \nDid not converge')
        else:
            print('\033[1m  Adam  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_adam = z_path
        self.z_adam = z
        return x_path,y_path,z_path
    
    
    def adamax(self, x_init, y_init, n_iter, lr, beta_1 = .9, beta_2 = .99, tol= 1e-5, epsilon = 1e-8):
        '''
        AdaMax is a variant of Adam with an infinity norm.

        Parameters
        ----------

        x_init: Start point at coordinate x as float

        y_init: Start point at coordinate y as float

        n_iter: Number of iteration as interger

        lr: Learning Rate given as float, size of learning rate affects each correction/step taken in descent

        beta_1: Float. Decay rate for the first moment estimates (mean)(m). Default = 0.9

        beta_2: Float. Decay rate for the second moment estimates (uncentered variance)(v). Default = 0.99

        tol: Tolerance rate to end descent, by Default = zero equivalent

        epsilon: Float constant. Used for numerical stability. Default = 10**-8


        Return
        ------

        Path of x, y and z taken to map the route of gradient descent. Returns as 3 one dimensional arrays.
        
        
        Reference
        ---------
        
        Source: http://ruder.io/optimizing-gradient-descent/index.html#adamax
                https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c

        '''

        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        m_x = 0
        v_x = 0
        m_y = 0
        v_y = 0

        lr_t = lr * (np.sqrt(1 - beta_2))/(1 - beta_1)

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)

            m_x = beta_1 * m_x + dx * (1 - beta_1)
            v_x = np.maximum(beta_2 * v_x, np.abs(dx))
            m_x_hat = m_x / (1 - np.power(beta_1, max(i, 1)))

            m_y = beta_1 * m_y + dy * (1 - beta_1)
            v_y = np.maximum(beta_2 * v_y, np.abs(dy))
            m_y_hat = m_y / (1 - np.power(beta_1, max(i, 1)))

            x += - (lr_t * m_x_hat) / v_x
            y += - (lr_t * m_y_hat) / v_y
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  AdaMax  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  AdaMax  \033[0m \nDid not converge')
        else:
            print('\033[1m  AdaMax  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_adamax = z_path
        self.z_adamax = z
        return x_path,y_path,z_path

    
    def nadam(self, x_init, y_init, n_iter, lr, beta_1 = .9, beta_2 = .99, tol= 1e-5, epsilon = 1e-8):
        '''
        Nesterov-accelerated Adaptive Moment Estimation (Nadam). Incorporating Nesterov Momentum into 
        Adam with the use of Nesterov momentum term for the first moving average.

        Parameters
        ----------

        x_init: Start point at coordinate x as float

        y_init: Start point at coordinate y as float

        n_iter: Number of iteration as interger

        lr: Learning Rate given as float, size of learning rate affects each correction/step taken in descent

        beta_1: Float. Decay rate for the first moment estimates (mean)(m). Default = 0.9

        beta_2: Float. Decay rate for the second moment estimates (uncentered variance)(v). Default = 0.99

        tol: Tolerance rate to end descent, by Default = zero equivalent

        epsilon: Float constant. Used for numerical stability. Default = 10**-8


        Return
        ------

        Path of x, y and z taken to map the route of gradient descent. Returns as 3 one dimensional arrays.
        
        
        Reference
        ---------
        
        Source: http://ruder.io/optimizing-gradient-descent/index.html#nadam
                https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c

        '''

        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        m_x = 0
        v_x = 0
        m_y = 0
        v_y = 0

        lr_t = lr * (np.sqrt(1 - beta_2))/(1 - beta_1)

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)

            m_x = beta_1 * m_x + (1 - beta_1) * dx
            v_x = beta_2 * v_x + (1 - beta_2) * np.power(dx, 2)
            m_x_hat = m_x / (1 - np.power(beta_1, max(i, 1))) + (1 - beta_1) * dx / (1 - np.power(beta_1, max(i ,1)))
            v_x_hat = v_x / (1 - np.power(beta_2, max(i, 1)))

            m_y = beta_1 * m_y + (1 - beta_1) * dy
            v_y = beta_2 * v_y + (1 - beta_2) * np.power(dy, 2)
            m_y_hat = m_y / (1 - np.power(beta_1, max(i, 1))) + (1 - beta_1) * dy / (1 - np.power(beta_1, max(i ,1)))
            v_y_hat = v_y / (1 - np.power(beta_2, max(i, 1)))
            

            x += - (lr_t * m_x_hat) / (np.sqrt(v_x_hat) + epsilon)
            y += - (lr_t * m_y_hat) / (np.sqrt(v_y_hat) + epsilon)
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  Nadam  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  Nadam  \033[0m \nDid not converge')
        else:
            print('\033[1m  Nadam  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_nadam = z_path
        self.z_nadam = z
        return x_path,y_path,z_path
    
    
    def amsgrad(self, x_init, y_init, n_iter, lr, beta_1 = .9, beta_2 = .99, tol= 1e-5, epsilon = 1e-8):
        '''
        AMSGrad is a variant of Adam that that uses the maximum of past squared gradients v. This memory
        application of AMSGrad with v(t-1) may solve specific convergence issues faced by adam. See sources 
        for details and proof
        
        Parameters
        ----------

        x_init: Start point at coordinate x as float

        y_init: Start point at coordinate y as float

        n_iter: Number of iteration as interger

        lr: Learning Rate given as float, size of learning rate affects each correction/step taken in descent

        beta_1: Float. Decay rate for the first moment estimates (mean)(m). Default = 0.9

        beta_2: Float. Decay rate for the second moment estimates (uncentered variance)(v). Default = 0.99

        tol: Tolerance rate to end descent, by Default = zero equivalent

        epsilon: Float constant. Used for numerical stability. Default = 10**-8


        Return
        ------

        Path of x, y and z taken to map the route of gradient descent. Returns as 3 one dimensional arrays.
        
        
        Reference
        ---------
        
        Source: http://ruder.io/optimizing-gradient-descent/index.html#amsgrad
                https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c

        '''

        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        m_x = 0
        v_x = 0
        v_x_hat = 0
        m_y = 0
        v_y = 0
        v_y_hat = 0

        lr_t = lr * (np.sqrt(1 - beta_2))/(1 - beta_1)

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
            dx, dy = self.fn_grad(self.fn, x, y)

            m_x = beta_1 * m_x + (1 - beta_1) * dx
            v_x = beta_2 * v_x + (1 - beta_2) * np.power(dx, 2)
            v_x_hat = max(v_x, v_x_hat)

            m_y = beta_1 * m_y + (1 - beta_1) * dy
            v_y = beta_2 * v_y + (1 - beta_2) * np.power(dy, 2)
            v_y_hat = max(v_y, v_y_hat)
            

            x += - (lr_t * m_x) / (np.sqrt(v_x_hat) + epsilon)
            y += - (lr_t * m_y) / (np.sqrt(v_y_hat) + epsilon)
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  AMSGrad  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  AMSGrad  \033[0m \nDid not converge')
        else:
            print('\033[1m  AMSGrad  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_amsgrad = z_path
        self.z_amsgrad = z
        return x_path,y_path,z_path
        
    
    def nag(self, x_init, y_init, n_iter, lr, beta = .9, tol= 1e-5):
        '''
        Nesterov accelerated gradient (nag) adds a directional element to momentum by calculating the 
        gradient from the approximate future position. Unlike momentum which progressively picks up 
        acceleration, Nag leaps in the direction of the previous accumulated gradient, measures and 
        corrects to update the nag. 

        Parameters
        ----------

        x_init: Start point at coordinate x as float. MUST BE FLOAT (e.g 1.0)

        y_init: Start point at coordinate y as float. MUST BE FLOAT (e.g 1.0)

        n_iter: Number of iteration as interger

        lr: Learning Rate given as float, size of learning rate affects each correction/step taken in descent

        beta: Float. Momentum term. Default = 0.9

        tol: Tolerance rate to end descent, by Default = zero equivalent


        Return
        ------

        Path of x, y and z taken to map the route of gradient descent. Returns as 3 one dimensional arrays.
        
        
        Reference
        ---------
        
        Source: http://ruder.io/optimizing-gradient-descent/index.html#nesterovacceleratedgradient

        '''

        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        nu_x = 0
        nu_y = 0

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
                
            mu = 1 - 3 / (i + 1 + 5) 
            dx, dy = self.fn_grad(self.fn, 
                                  x - mu * nu_x,
                                  y - mu * nu_y)
            
            nu_x = beta * nu_x + lr * dx
            nu_y = beta * nu_y + lr * dy

            x += - nu_x
            y += - nu_y
            
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  Nesterov accelerated gradient  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  Nesterov accelerated gradient  \033[0m \nDid not converge')
        else:
            print('\033[1m  Nesterov accelerated gradient  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_nag = z_path
        self.z_nag = z
        
        return x_path,y_path,z_path
    

    def RMSprop(self, x_init, y_init, n_iter, lr = 0.001, beta = .9, tol= 1e-5, epsilon = 1e-8):
        '''
        RMSprop divides the learning rate by an exponentially decaying squared gradient

        Parameters
        ----------

        x_init: Start point at coordinate x as float. MUST BE FLOAT (e.g 1.0)

        y_init: Start point at coordinate y as float. MUST BE FLOAT (e.g 1.0)

        n_iter: Number of iteration as interger

        lr: Learning Rate given as float, size of learning rate affects each correction/step taken in descent. Default = 0.001

        beta: Float. Momentum term. Default = 0.9

        tol: Tolerance rate to end descent, by Default = zero equivalent
        
        epsilon: Float constant. Used for numerical stability. Default = 10**-8


        Return
        ------

        Path of x, y and z taken to map the route of gradient descent. Returns as 3 one dimensional arrays.
        
        
        Reference
        ---------
        
        Source: http://ruder.io/optimizing-gradient-descent/index.html#rmsprop

        '''

        x, y = x_init,y_init
        z = self.fn(x, y)

        x_path = []
        y_path = []
        z_path = []

        x_path.append(x)
        y_path.append(y)
        z_path.append(z)

        dx, dy = self.fn_grad(self.fn, x, y)
        
        dx_sq = dx**2
        dy_sq = dy**2

        for i in range(n_iter):
            if np.abs(dx) < tol or np.isnan(dx) or np.abs(dy) < tol or np.isnan(dy):
                break
                
            dx, dy = self.fn_grad(self.fn, x, y)
            
            dx_sq = beta * dx_sq + (1 - beta) * dx * dx
            dy_sq = beta * dy_sq + (1 - beta) * dy * dy

            x += - lr * dx / np.sqrt(dx_sq + epsilon)
            y += - lr * dy / np.sqrt(dy_sq + epsilon)
            
            x_path.append(x)
            y_path.append(y)
            z = self.fn(x, y)
            z_path.append(z)

        if np.isnan(dx) or np.isnan(dy):
            print('\033[1m  RMSprop  \033[0m \nExploded')
        elif np.abs(dx) < tol and np.abs(dy) < tol:
            print('\033[1m  RMSprop  \033[0m \nDid not converge')
        else:
            print('\033[1m  RMSprop  \033[0m \nConverged in {} steps.  \nLoss fn {:0.4f} \nAchieved at coordinates x,y = ({:0.4f}, {:0.4f})'.format(i, z, x, y))

        self.z_path_RMSprop = z_path
        self.z_RMSprop = z
        
        return x_path,y_path,z_path
    
    
