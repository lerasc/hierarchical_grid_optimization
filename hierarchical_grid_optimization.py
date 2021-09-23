
import numpy as np
import multiprocessing

from math      		import log
from numbers   		import Number 
from functools 		import partial
from itertools 		import product
from numpy.random 	import choice


def grid_optimization(  objfunc,
						bounds,
						discretizations,
						constraint 			=  None,
						scale 				= 'linear',
						base 				=  10,
						minimize          	=  True,
						max_eval 			=  None,
						failsave          	=  False,
						parallel			=  False,
						full_ret	        =  False,
						):
	"""
	Return value is the value x at which objfunc(x) is minimal (or maximal). This value is determined from a simple
	brute force search over a grid. Multiple grid-layers are possible, meaning that the initial grid is refined around
	the minimum point, another minimum point is found, the neighborhood of that minimum is again refined, and so on.
	See example below.

	input:
	-----
	:param objfunc:			Function of one argument x (potentially vectorial) that we want to minimize (or maximize).

	:param bounds:			List of tuples, where each tuple is of the form (minval,maxval) specifying the bounds
							between which we discretize the value range. One tuple per objfunc argument, i.e. len(x)
							tuples.

	:param discretizations:	Dictionary with keys 1,2,3...,n specifying the number of discretization points for the n
							layers.	Corresponding value is a list that specifies the number of discretization points for
							each of the parameters. See example below.

	:param constraint:		A obj that takes as input the objfunc argument x, and returns a bool. Returns True if the
							constraint is satisfied and False if it is violated. As such, the constraint obj is very
							primitive as it still creates an entire parameter grid, irrespective of the constraint. But
							if the constraint is not met, the obj value is set to +infinity, so it is defenitely not
							selected as a minimum.

	:param scale:			Either 'linear' or 'log', specifying whether the grid points between minval and maxval
							should be partitioned into equidistant points on a linear or logarithmic scale. Can also be
							a list of size len(x), specifying for each argument whether it is 'linear' or 'log'

	:param base:			Only used if scale='log'. Specifies the base of the logarithm.

	:param minimize:		If True, find the minimum of the objfunc. If False, find the maximum.

	:param max_eval:		If x is high-dimensional, it quickly becomes infeasable to evaluate the obj on a dense grid
							(curse of dimensionality). Thus, one can restrict the obj to evaluate objfunc on only
							max_eval randomly selected grid-points on each layer. This restricts the obj to a total of
							n*max_eval evaluations, and turns the obj into a stochastic brute-force method.

	:param failsave:		If True, NaN is returned for every obj call objfunc(x) that raises an error. This is useful
							to avoid the entire program from crashing, if objfunc happens to not be defined everywhere.

	:param parallel:		If True, then the multiprocessing library is used. Number of available kernels are detected
							auotmatically.

	:param full_ret:		If True, return not only the optimum, but also an array of dimension (T, K+1) which contains
							all the objective evaluations. Here, T is the number of grid-points that have been evaluated
							and K is the number of variables of the objective function. The last column represents the
							objective value, while the first K columns are the grid points.

	output:
	------
	:return x:  			value of grid point at which objfunc is minimal (or maximal).

	example:
	-------
	Assume
	>> objfunc = lambda x: (x[0]-2)**2 + (x[1]-100)**2
	We look for the minimum, which is found at (2, 100), within the following bounds:
	>> bounds = [ (-5,5), (50,150)   ]
	Assume we want to refine the grid 4 times, and we decrease the grid granularity in each step. 
	A possible argument for discretizations is then: 
	>> discretizations = 	{
							1: [ 100, 100 ],	# evaluate objfunc on a on 100 x 100 grid and find minimum
							2: [ 50,  60], 		# split refined grid into into 50 x 60 points around previous minimum
							3: [ 20,  30],		# then a grid of 20 x 30 points around previous minimum
							4: [ 4,  4],		# finally only a 4 x 4 grid around previous minimum
							}
	A special case is when at some point we want to keep a parameter fixed. In that case, just set that discretization 
	range to 1. This will keep the paramter fixed until the	end of the optimization.
	"""

	# make sure provided input is correct
	####################################################################################################################
	assert callable(objfunc), 'objfunc must be obj'

	for t in bounds: assert( len(t)==2 ), 'bounds must be list of (min,max) tuples'

	dict_args 	= list(np.sort(list(discretizations.keys()))) 	# sorted dict arguments
	should_args = list(np.arange(1,len(discretizations)+1)) 	# what the keys should be
	assert dict_args==should_args, 'invalid discretization key arguments, should be 1,2,...'
	for i, vals in discretizations.items():
		assert( len(vals)==len(bounds) ), 'discretization arrays not of same size as bounds'

	if constraint is not None: assert callable(constraint), 'constraint must be a obj'
	assert isinstance(base,Number), 'base must be number'

	assert minimize in [True,False], 'minimize must be a bool'

	if max_eval is not None: assert isinstance(max_eval,int), 'max_eval must be None or integer'


	# initializations
	####################################################################################################################
	evals = [] 																# stores all evaluated objective values
	dim   = len(bounds) 													# number of parameters
	bnds  = bounds.copy() 													# because it will be overwritten below
	bnds  = [(float(lb), float(up)) for (lb,up) in bnds] 					# replace by floats to avoid problems
	N 	  = len(discretizations)											# number of grid layers
	fnct  = partial( obj_evaluation,										# dont use lambda (cannot be pickled)
					obj         = objfunc,
					constraint  = constraint,
					mimimize    = minimize,
					failsave    = failsave,
					)


	# adjust the scale argument for consistent internal use
	####################################################################################################################
	if isinstance(scale,str): scale = [scale for _ in bnds ] # one scale argument per dimension of x
	assert isinstance(scale,list) and len(scale)==len(bnds),"scale must be of same length as bounds"
	for s in scale: assert s in ['linear','log'],"scale must be 'linear' or 'log'."

	# refine grid some N times
	####################################################################################################################
	for l in np.arange(1,N+1): # iterate over all levels (dont iterate over dict keys, sorting not guarantueed)

		# for each dimension (parameter), store grid points
		################################################################################################################
		gridpoints = []
		for (lower,upper),discretization,sc in zip(bnds,discretizations[l],scale):

			if sc=='linear': 	gridline = np.linspace(lower, upper, discretization )
			else:				gridline = np.logspace(log(lower,base), log(upper,base), base=base, num=discretization)
			gridpoints += [ gridline ]

		# form all parameter combinations, i.e. all grid points to evaluate
		################################################################################################################
		if max_eval is None: 
			if dim==1: 									# special case: 1d optimization problem			
				products = list(gridpoints[0])
			else: 										# generic case
				products = list(product(*gridpoints)) 	# all combinations

		# if generating and/or evaluation all combinations is too costly, just select a subset of them
		################################################################################################################
		else:
			products    	= []			 	 										# stores subset of gridpoints
			nvals       	= np.prod([len(gridline) for gridline in gridpoints])		# total number of gridpoints
			if nvals < 0:	  nvals = max_eval 											# can happen due to overflow
			nsel			= min( max_eval, nvals )									# nr of gripoints to be selected

			for _ in range(nsel):														# generate nsel gridpoints
				gp             = tuple([ choice(gridline) for gridline in gridpoints ])	# one such gridpoint
				if dim==1: gp  = gp[0]													# special case: make number
				products      += [ gp ] 												# add to the list 

		# evaluate all grid points 
		################################################################################################################
		if parallel:

			nc   = multiprocessing.cpu_count() 			# number of cores available
			pool = multiprocessing.Pool(processes=nc) 	# initialize multiprocessing instance
			vals = list(pool.map(fnct,products)) 		# evaluate obj on all grid points in parallel
			pool.close()								# close multiprocessing instance
			pool.join()									# close multiprocessing instance

		else:

			vals = [fnct(p) for p in products]			# evaluate obj on all grid points sequentially

		# store all the evaluations
		################################################################################################################
		if dim==1:  concat   = np.array(list(zip(products,vals))) 					   # concatenate param and objective
		else: 		concat   = np.array([ list(p)+[v] for p,v in zip(products,vals) ]) # concatenate param and objective
		evals  				+= [concat]											       # append to list

		# bad case: all nans over the entire grid
		################################################################################################################
		if np.sum(np.isnan(vals))==len(vals):

			dummy_ret = len(bounds)*[np.nan]

			if full_ret: return dummy_ret, np.vstack(evals)
			else:		 return dummy_ret

		# generic case: we have non-nan values, so we determine the minimum/maximum
		################################################################################################################
		find_opt_arg = np.nanargmin if minimize else np.nanargmax 		# whether to find argmin or argmax
		if dim==1: 	optarg = [ products[find_opt_arg(vals)] ]
		else: 		optarg = list( products[ find_opt_arg(vals) ] )

		# if there is a next refinement, find new grid boundaries and refine the grid for another round of evaluations
		################################################################################################################
		if l < N:

			# determine new grid by finding left and right neighbors of minimum point
			############################################################################################################
			for i,xopt in enumerate(optarg): # iterate over all dimensions, i.e. obj arguments

				gridline      = gridpoints[i] 				# all 1d grid points for that argument
				optimum_index = list(gridline).index(xopt) 	# find index of minimum argument

				# special case: we have already fixed that parameter at this level, so it remains fixed
				if discretizations[l][i]==1:
					new_lower = new_upper = gridline[0] 	# = gridline[optimum_index], because it is the only one

				# special case: we want to have only one point at next level
				elif discretizations[l+1][i]==1:
					new_lower = new_upper = gridline[optimum_index]

				# special case: we are at the left boundary
				elif optimum_index==0:
					new_lower = gridline[0]
					new_upper = gridline[1]

					if l==1: print(f'Attention, optimum in first layer falls on left boundary in dimension {i}.')

				# special case: right boundary
				elif optimum_index==discretizations[l][i]-1:
					new_lower = gridline[-2]
					new_upper = gridline[-1]

					if l==1: print(f'Attention, optimum in first layer falls on right boundary in dimension {i}.')

				# generic case: minimum is somewhere in the middle
				else:
					new_lower = gridline[optimum_index-1]
					new_upper = gridline[optimum_index+1]
				
				bnds[i] = (new_lower,new_upper) # update the search boundaries

	if full_ret: return optarg, np.vstack(evals)
	else:		 return optarg


def obj_evaluation( x, obj, constraint=None, mimimize=True, failsave=True ):
	"""
	Subroutine of grid_optimization that evaluates the objective obj obj at value x. It just returns obj(x), but does it
	in a safe way: if an error is triggered during the evaluation of obj(x) and if failsave=True, then a NaN is returned
	to avoid the entire optimization scheme from crashing. Second, if a constraint is provided, then args that violate
	these constraints trigger np.inf as return value (-np.inf if minimize=False).
	"""

	if constraint is not None: 								# if a constraint was provided...
		if constraint(x) is False:							# ...and if that constraint is violated...
			return np.inf if mimimize else -np.inf			# ...then return inf if we look for minima, else -np.inf

	if failsave:											# if we want to avoid errors
		try: 	return obj(x)								# try to evaluate the obj
		except: return np.nan 								# if it doesnt work, set it to NaN

	return obj(x) 											# evaluate and return
