import casadi as ca

from dynamics import casadi_3d_dynamics as dynamics

def check_dynamics():
    x = ca.MX.sym('x', 13)
    u = ca.MX.sym('u', 3)

    xdot = dynamics(x, u)
    f = ca.Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])
    print(f)
    x_test = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    u_test = [0, 0, 1]
    print(f(x_test, u_test))



def check_integrator():
    x = ca.MX.sym('x', 11)
    u = ca.MX.sym('u', 3)

    xdot = dynamics(x, u)
    f = ca.Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])

    # Integrator to discretize the system
    intg_options = {
        "tf": 0.1,  # timestep
        "simplify": True,
        "number_of_finite_elements": 4
    }

    # DAE problem structure
    dae = {
        "x": x,         # What are states?
        "p": u,         # What are parameters (=fixed during the integration horizon)?
        "ode": f(x,u)   # Expression for the right-hand side
    }

    intg = ca.integrator('intg','rk', dae, intg_options)

if __name__ == '__main__':
    check_dynamics()