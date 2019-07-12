from IPython.display import Latex, display

def display_J_mu():
    display(Latex(r'''${\bf J}_{\mu} = 
                            \begin{bmatrix}
                            {\bf O} & {\bf O}\\
                            {\bf I} & -{\bf I}
                            \end{bmatrix}\\
                            \mu = 
                            \begin{bmatrix}
                            \mu^{\max}\\
                            \mu^{\min}
                            \end{bmatrix}
                            $'''))
    
def display_C_gamma():
    display(Latex(r'''$
                                {\bf C} = 
                                \begin{bmatrix}
                                {\bf C}_p\\
                                -{\bf C}_d
                                \end{bmatrix}\\
                                \gamma^{\max/\min} = 
                                \begin{bmatrix}
                                \pi^{\max/\min}\\
                                \psi^{\max/\min}
                                \end{bmatrix}\\
                                {\cal I}_\gamma = \{i|~i \text{ is not price forming}\}
                                $'''))

def ProblemInfo(options):
    if options['mode'] == separate_mode:
        print('SEPARATE SOLUTION')
        if options['lam_q'] == lam_q_fixed:
            print('System 1')
            display_J_mu()
            display(Latex(r'''$
                    {\bf A} = 
                    \begin{bmatrix}
                    {\bf J}_{P~ok}^T & {\bf J}_S^{T} & {\bf J}_{\mu}
                    \end{bmatrix}~~~
                    {\bf x} = 
                    \begin{bmatrix}
                    \lambda_{ok}^P\\
                    \sigma\\
                    \mu
                    \end{bmatrix}~~~
                    {\bf b} = - {\bf J}_Q^T\lambda^Q - {\bf J}_{P~form}^T\lambda_{form}^{P~new}
                    $'''))
            print('System 2')
            display_C_gamma()
            
            display(Latex(r'''\begin{equation}
                                    {\bf B} = 
                                    \begin{bmatrix}
                                    -{\bf I} & {\bf I}
                                    \end{bmatrix}~~~
                                    {\bf y} = 
                                    \begin{bmatrix}
                                    \gamma^{\max}[{\cal I}_\gamma]\\
                                    \gamma^{\min}[{\cal I}_\gamma]
                                    \end{bmatrix}~~~
                                    {\bf c} = {\bf C[{\cal I}_\gamma]} + {\rm sign}({\bf C[{\cal I}_\gamma]})\lambda^P[{\cal I}_\gamma]
                                \end{equation}'''))
        else:
            pass
    else:
        print('SIMULTANEOUS SOLUTION')
        display_J_mu()
        display_C_gamma()
        display(Latex(r'''\lambda^P = -\gamma^{\max} + \gamma^{\min} - {\rm abs}({\bf C})'''))