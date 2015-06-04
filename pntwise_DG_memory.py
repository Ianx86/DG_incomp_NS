from dolfin import *
import numpy as np

# Problem parameters
####################
dim = 2

T_end		= 0.5
dt		= 0.1
t 		= dt # time variable

Ns		= 15 # number of memory modes on [0,+infty)
s_nodes		= Constant([0,0.1,0.2,0.3,0.4,0.5,0.8,1.5,2.2,3.0,4.0,5.8,7.5,10.0,20.0])

u0		= Expression(("0.0", "0.0"), t = 0.0)
f_rhs 		= Expression(("0.0", "0.0"), t = 0.0)
G		= Expression("exp(-x[0])")

sigma		= 10.0

atol, rtol = 1e-7, 1e-10      # abs/rel tolerances
lmbda      = 1.0              # relaxation parameter
residual   = 1                # residual L2-norm error
residual_1 = 1		      # residual from previous Newton iteration
rel_res    = residual         # relative residual error
maxIter    = 10 


# Mesh
######
Right	= 5.0
Top	= 1.0
mesh 	= RectangleMesh(0.0,0.0,Right,Top, 50, 10,"crossed")
bndry 	= FacetFunction("size_t", mesh)

# Boundary description
######################
class LeftBoundary(SubDomain):
	def inside(self, x, bndry):
		return x[0] < DOLFIN_EPS

class RightBoundary(SubDomain):
	def inside(self, x, bndry):
		return x[0] > Right-DOLFIN_EPS

class TopBoundary(SubDomain):
	def inside(self, x, bndry):
		return x[1] > Top-DOLFIN_EPS

class BottomBoundary(SubDomain):
	def inside(self, x, bndry):
		return x[1] < DOLFIN_EPS

# Function spaces
#################
S = FunctionSpace(mesh,'DG',1)
V = VectorFunctionSpace(mesh, 'DG', 1)
Q = FunctionSpace(mesh, 'DG', 0)
VQ = MixedFunctionSpace([V,Q])
W = TensorFunctionSpace(mesh, 'DG', 1, shape=(dim,dim))
WW = TensorFunctionSpace(mesh, 'DG', 1, shape=(dim,dim,Ns))


# Boundary conditions
#####################
noslip	  = Expression(("0.0", "0.0"), t = 0.0)
u_IN	  = Expression(("x[1]*(1.0-x[1])", "0.0"), t = 0.0)

bc_top	  = DirichletBC(VQ.sub(0), noslip, TopBoundary(), 'pointwise')
bc_bottom = DirichletBC(VQ.sub(0), noslip, BottomBoundary(), 'pointwise')
bc_left   = DirichletBC(VQ.sub(0), u_IN, LeftBoundary(), 'pointwise')

bcs 	= [bc_top, bc_bottom, bc_left]
bcs_hom = homogenize(bcs) # residual is zero on boundary

bcB 	= []

# Unknown and test functions
############################
w 	= Function(VQ)
w_inc 	= Function(VQ) # residual for Newton method
(u, p)  = split(w)
(v, q)  = TestFunctions(VQ)
u_1	= Function(V) # velocity from previous time step
u_nPrev	= Function(V) # velocity from previous Newton iteration
dup	= TrialFunction(VQ)

B	= Function(WW) # gather functions from each s-node
B_1	= Function(WW) # B from previous t-step

# Memory init
#############

# compute the weights given by memory function G
weights		= np.zeros(Ns-1)
i 		= 0
vS 		= np.zeros(Ns) # convert s_nodes to array of floats
tau_k		= np.zeros(Ns-1)
s_nodes.eval(vS, np.zeros(Ns+1))

while (i < Ns-1):
	tau_k[i]	= vS[i+1] - vS[i]
	memoryInt	= IntervalMesh(10,vS[i],vS[i+1])
	weights[i]	= assemble(G*dx(memoryInt))
	print weights[i]	
	i = i + 1

# prepare B
B_old = Function(W)
B_old = interpolate(Expression((("1", "0"), 
                             ("0", "1"))), W)
for K in range(Ns):
	assign([B.sub(K+i*Ns) for i in range(dim*dim)], B_old)

# Variational formulation of motion
###################################
n 	= FacetNormal(mesh)
ds	= Measure("ds", subdomain_data = bndry)
I 	= Identity(u.geometric_dimension())
D 	= 0.5*(grad(u) + grad(u).T)

def F(D):
	F = D*D
	return F

def der_F(D):
	F = 2*D
	return D

def lin_F(D,u_1):
	D_1 = 0.5*(grad(u_1) + grad(u_1).T)
	M = F(D_1) + der_F(D_1)*(D-D_1)
	return D

def a(u,v):
	M = inner(lin_F(D,u_1),grad(v))*dx - inner(avg(lin_F(D,u_1))*n('+'),jump(v))*dS 
	return M

def J(u,v):
	M = sigma*inner(jump(u),jump(v))*dS
	return M

def b(p,v):
	M = -p*div(v)*dx  + avg(p)*dot(jump(v),n('+'))*dS
	return M

def c(u,w,v):	
	P = avg(dot(w,n))
	H = conditional(P < 0.0, dot(u('+'),w('+')), dot(u('-'),w('-')))
	M = -0.5*inner(grad(v)*u,w)*dx + inner(0.5*H*n('+'),jump(v))*dS
	return M

def L(v,B):
	j = 0
	mem = Function(W)
	Bt = Function(W)
	while (j < Ns-1):
		assign(Bt, [B.sub(j+i*Ns) for i in range(dim*dim)])
		mem.vector().axpy(weights[j], Bt.vector())
		j = j+1
	M = inner(f_rhs,v)*dx - inner(mem,grad(v))*dx + inner(avg(mem)*n('+'),jump(v))*dS
	return M

T 	= (1/dt)*inner(u - u_1,v)*dx + a(u,v) + J(u,v) + b(p,v) + b(q,u) + c(u,u_nPrev,v) - L(v,B) 

Jac 	= derivative(T, w, dup)

# Variational formulation for memory
####################################

def BgradU(B_s,u,i,j):
	M = Function(S)					
	for k in range(dim):
		M += u[i].dx(k) * B_s[k,j]	
	return M

def BgradUT(B_s,u,i,j):
	M = Function(S)
	for k in range(dim):
		M += u[j].dx(k) * B_s[i,k]
	return M

def UgradB(B_s,u,i,j):
	M = Function(S)
	for k in range(dim):
		M += u[k]*B_s[k,i].dx(j)
	return M

def UPwind(B_s,u,n,i,j,b_):
	P = avg(dot(u,n))
	H = conditional(P < 0.0, dot(u('+'),n('+')), 0.0)
	M = H*jump(B_s[i,j])*b_('+')*dS
	return M


def form_Bform(dt,tau,alpha_0,alpha_1,alpha_2,i,j,bt,Bs0,Bs1,Bs2,b_,u,n):
	Bform = tau*bt*b_*dx - tau*Bs0[i,j]*b_*dx + dt*alpha_0*Bs0[i,j]*b_*dx + dt*alpha_1*Bs1[i,j]*b_*dx + dt*alpha_2*Bs2[i,j]*b_*dx + dt*tau*UgradB(Bs0,u,i,j)*b_*dx - dt*tau*BgradU(Bs0,u,i,j)*b_*dx - dt*tau*BgradUT(Bs0,u,i,j)*b_*dx - dt*tau*UPwind(Bs0,u,n,i,j,b_) 
	return Bform

# Definition of solve procedures
################################

def solve_deformation(i,B_1,B,u,n,dt,tau_k):	
	B_s0	= Function(W)
	B_s1	= Function(W)
	B_s2	= Function(W) # B from previsou t-step in s_i, s_i-1 and s_i-2
	b	= Function(S)
	b_	= TestFunction(S)
	b_trial	= TrialFunction(S)
	
	assign(B_s0, [B_1.sub(i+j*Ns) for j in range(dim*dim)])
	if (i > 0):
		assign(B_s1, [B_1.sub(i-1+j*Ns) for j in range(dim*dim)])
	else:
		B_s1 = project(I,W)
	if (i > 1):
		assign(B_s2, [B_1.sub(i-2+j*Ns) for j in range(dim*dim)])	
	else:
		B_s2 = project(I,W)
	
	alpha_0 = 1 # BDF koef. je treba doplnit vypocet
	alpha_1 = 1
	alpha_2 = 1
	
	for k in range(dim):
		for l in range(dim):
			Bform = form_Bform(dt,tau_k[i],alpha_0,alpha_1,alpha_2,k,l,b,B_s0,B_s1,B_s2,b_,u,n)
			JacB = derivative(Bform,b,b_trial)
			problem = NonlinearVariationalProblem(Bform, b, bcB, JacB)
			solver = NonlinearVariationalSolver(problem)
			solver.solve()

			Bind = k*dim + l
			assign(B.sub(i+Bind*Ns) , b) # saves the result


def solve_motion(fenicsNewton,T,w,bcs,bcs_hom,Jac):
	if fenicsNewton:
		problem = NonlinearVariationalProblem(T, w, bcs, Jac)
		solver = NonlinearVariationalSolver(problem)
		solver.solve()
	else:
		nIter = 0
		residual = 1
		rel_res = 1
		while residual > atol and rel_res > rtol and nIter < maxIter:     # Newton iterations
			nIter += 1
        		A, b = assemble_system(Jac, -T, bcs_hom)
        		solve(A, w_inc.vector(),b)     # Determine step direction
			residual = b.norm('l2')
			if (nIter == 1):
				rel_res = 1
			else:
				rel_res = residual/residual_1
			residual_1 = residual
			string = "Newton iteration %d: r (abs) = %.3e (tol = %.3e) r (rel) = %.3e (tol = %.3e)"
			print string % (nIter, residual, atol, rel_res, rtol)        		      		
        		w.vector()[:] += lmbda*w_inc.vector()    # New w vector			
			#u_nPrev.assign(u)  
	



# Solution
##########

plot(mesh,interactive=True)

ufile = File("u.xdmf")
pfile = File("p.xdmf")
u_1.rename("u", "velocity")
ufile << u_1

plt = plot(u, mesh = mesh, mode="color", interactive = True, wireframe = False)
pltP = plot(p, mesh = mesh, mode="color", interactive = True, wireframe = False)

u_1.assign(interpolate(u0,V)) # apply initial condition
u_nPrev.assign(interpolate(u0,V))
for bc in bcs:
	bc.apply(w.vector())  # apply boundary condition


while t<=T_end:
	noslip.t 	= t
	u_IN.t		= t
	f_rhs.t		= t
	print 'time step: ', t

	print 'STARTING THE COMPUTATION OF MOTION'
	solve_motion(True,T,w,bcs,bcs_hom,Jac)
	(u,p) = w.split(True)
	print 'ENDING THE COMPUTATION OF MOTION'

	print 'STARTING THE COMPUTATION OF DEFORMATION'
	assign(B_1,B) # new B will be computed
	i = 0	
	while (vS[i] <= t and i < Ns-1)
		solve_deformation(i,B_1,B,u,n,dt,tau_k)
		i += 1
	print 'ENDING THE COMPUTATION OF DEFORMATION'

	u_1.assign(u)
	u_nPrev.assign(u)
		
	u.rename("u", "velocity")
	p.rename("p", "pressure")
	ufile << (u,t)
	pfile << (p,t)
	t += float(dt)


plt.plot(u)
pltP.plot(p)
	


 



