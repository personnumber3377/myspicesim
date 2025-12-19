
from scipy.sparse import csr_matrix
import numpy as np
import copy
import matplotlib.pyplot as plt

def generate_matrix(indexes, values, shape=None):
	if shape == None:

		max_row_num = max([thing[0] for thing in indexes])
		max_col_num = max([thing[1] for thing in indexes])
		matrix = [[0.0 for _ in range((max_col_num+1))] for _ in range(max_row_num+1)]
	else:
		if shape[0] != shape[1]:
			print("Error: Given shape to generate_matrix must be square! Now it is this : "+str(shape))
			exit(1)
		#matrix = [[0.0 for _ in range((shape[0]+1))] for _ in range(shape[1]+1)]
		matrix = [[0.0 for _ in range((shape[0]))] for _ in range(shape[1])]
	count = 0

	for ind in indexes:

		matrix[ind[0]][ind[1]] += values[count]
		count += 1
	if shape == None:
		if max_row_num != max_col_num:
			print("Number of rows in matrix must be same as number of columns aka matrix must be square!")
			exit(1)
	
		return matrix, max_row_num
	else:
		return matrix, len(matrix[0])

def flatten(S):
	if S == []:
		return S
	if isinstance(S[0], list):
		return flatten(S[0]) + flatten(S[1:])
	return S[:1] + flatten(S[1:])

def get_max_node_num(all_nodes):
	return max(flatten(all_nodes))

def get_capacitance_and_inductance_matrix(nodes_inductors, inductor_values, nodes_capacitors, capacitor_values, max_nodes, g_shape):

	# This basically just returns the C matrix.

	g = []
	g_row = []
	g_col = []

	nodes_inductors = sorted(nodes_inductors)
	
	for index, ind_val in enumerate(inductor_values):
		# just put this on the diagonal and you should be fine. then there will be equations like v_a - L*d(I_a(t))/dt = 0 etc etc.

		g.append(-1*ind_val)
		g_col.append(max_nodes+index)
		g_row.append(max_nodes+index)

	# go through capacitors:

	#for index, cap_val in enumerate(capacitor_values):

	# These are added to the current equations, because for a capacitor the equation I(t) = C*(d(V(t))/dt) holds
	print("nodes_capacitors == "+str(nodes_capacitors))
	for index, cap_val in enumerate(nodes_capacitors):
		N1 = cap_val[0]
		N2 = cap_val[1]

		print("N1: "+str(N1))
		print("N2: "+str(N2))
		print("values[ir]: "+str(capacitor_values[index]))
		if N1 == 0 or N2 == 0:

			g.append(1*capacitor_values[index])
			g_row.append(max(N1, N2) - 1)
			g_col.append(max(N1, N2) - 1)
		
		else:
			# This adds the admittance to the start point.
			g.append(1*capacitor_values[index])
			g_row.append(N1 - 1)
			g_col.append(N1 - 1)
			# This adds the admittance to the start point.
			g.append(1*capacitor_values[index])
			g_row.append(N2 - 1)
			g_col.append(N2 - 1)

			g.append(-1*capacitor_values[index])
			g_row.append(N1 - 1)
			g_col.append(N2 - 1)

			g.append(-1*capacitor_values[index])
			g_row.append(N2 - 1)
			g_col.append(N1 - 1)

	#resulting_matrix = generate_matrix()
	index_stuff = [[g_row[i],g_col[i]] for i in range(len(g_col))]

	print("Index stuff in get_capacitance_and_inductance_matrix: "+str(index_stuff))
	print("g in get_capacitance_and_inductance_matrix: "+str(g))

	resulting_matrix, max_shit = generate_matrix(index_stuff, g, shape=g_shape)


	poopoothing = flatten([nodes_inductors, nodes_capacitors])
	print("poopoothing: "+str(poopoothing))
	poopoothing = max(poopoothing)

	stuff = csr_matrix((g,(g_row, g_col)), shape=tuple(g_shape))
	print(stuff)
	print(stuff.toarray())


	print("Resulting matrix: "+str(resulting_matrix))
	return resulting_matrix, poopoothing

def get_conductance_matrix(resistor_values, nodes_resistors, nodes_voltages, voltage_values, nodes_inductors):
	
	max_nodes = get_max_node_num(nodes_resistors+nodes_voltages+nodes_inductors)

	g = []
	g_row = []
	g_col = []
	count = 0
	print("========================================================")
	for node in nodes_resistors:
		N1 = node[0]
		N2 = node[1]

		print("N1: "+str(N1))
		print("N2: "+str(N2))
		print("values[ir]: "+str(resistor_values[count]))
		if N1 == 0 or N2 == 0:

			g.append(1/resistor_values[count])
			g_row.append(max(N1, N2) - 1)
			g_col.append(max(N1, N2) - 1)
		
		else:
			# This adds the admittance to the start point.
			g.append(1/resistor_values[count])
			g_row.append(N1 - 1)
			g_col.append(N1 - 1)
			# This adds the admittance to the start point.
			g.append(1/resistor_values[count])
			g_row.append(N2 - 1)
			g_col.append(N2 - 1)

			g.append(-1/resistor_values[count])
			g_row.append(N1 - 1)
			g_col.append(N2 - 1)

			g.append(-1/resistor_values[count])
			g_row.append(N2 - 1)
			g_col.append(N1 - 1)
		count += 1
		print("g : "+str(g))
		print("g_row: "+str(g_row))
		print("g_col: "+str(g_col))
	print("========================================================")

	if len(g_col) != len(g_row): # the matrix must be square
		print("Somemax_nodes went wrong.")
		exit(1)
	print("g : "+str(g))
	print("g_row: "+str(g_row))
	print("g_col: "+str(g_col))
	print("resistor_values: "+str(resistor_values))

	for k, il in enumerate(nodes_inductors):

		# this is basically just a copy of the voltage source max_nodes, this is because inductors are identical to batteries/independent voltage sources with zero voltage. Because inductors are basically a short.


		N1 = il[0]
		N2 = il[1]

		if N1 == 0:
			print("Voltage source N1 grounded.")
			# negative terminal
			g.append(-1)
			g_row.append(N2 - 1)
			g_col.append(max_nodes + len(nodes_inductors) + k)

			# negative terminal
			g.append(-1)
			g_row.append(max_nodes + len(nodes_inductors) + k)
			g_col.append(N2 - 1)
		elif N2 == 0:
			print("Voltage source N1 grounded.")
			# negative terminal
			g.append(1)
			g_row.append(N1 - 1)
			g_col.append(max_nodes + len(nodes_inductors) + k)

			# negative terminal
			g.append(1)
			g_row.append(max_nodes + len(nodes_inductors) + k)
			g_col.append(N1 - 1)
		else:
			# positive terminal
			g.append(1)
			g_row.append(N1 - 1)
			g_col.append(max_nodes + len(nodes_inductors) + k)

			# positive terminal
			g.append(1)
			g_row.append(max_nodes + len(nodes_inductors) + k)
			g_col.append(N1 - 1)

			# negative terminal
			g.append(-1)
			g_row.append(N2 - 1)
			g_col.append(max_nodes + len(nodes_inductors) + k)

			# negative terminal
			g.append(-1)
			g_row.append(max_nodes + len(nodes_inductors) + k)
			g_col.append(N2 - 1)
		#oofmax_nodes += 1

	print("max_nodes: "+str(max_nodes))

	for k, iv in enumerate(nodes_voltages):

		N1 = iv[0]
		N2 = iv[1]

		if N1 == 0:
			print("Voltage source N1 grounded.")
			# negative terminal
			g.append(-1)
			g_row.append(N2 - 1)
			g_col.append(max_nodes + k)

			# negative terminal
			g.append(-1)
			g_row.append(max_nodes + k)
			g_col.append(N2 - 1)
		elif N2 == 0:
			print("Voltage source N1 grounded.")
			# negative terminal
			g.append(1)
			g_row.append(N1 - 1)
			g_col.append(max_nodes + k)

			# negative terminal
			g.append(1)
			g_row.append(max_nodes + k)
			g_col.append(N1 - 1)
		else:
			# positive terminal

			print("max_nodes+k : " +str(max_nodes+k))
			print("N1 :"+str(N1))
			print("N2 :"+str(N2))
			g.append(1)
			g_row.append(N1 - 1)
			g_col.append(max_nodes + k)

			# positive terminal
			g.append(1)
			g_row.append(max_nodes + k)
			g_col.append(N1 - 1)

			# negative terminal
			g.append(-1)
			g_row.append(N2 - 1)
			g_col.append(max_nodes +0 + k)

			# negative terminal
			g.append(-1)
			g_row.append(max_nodes + 0 + k)
			g_col.append(N2 - 1)

	print("Final stuff: ")
	print("g_row: "+str(g_row))
	print("g_col: "+str(g_col))
	print("g: "+str(g))
	index_stuff = [[g_row[i],g_col[i]] for i in range(len(g_col))]
	
	resulting_matrix, max_shit = generate_matrix(index_stuff, g)

	stuff = csr_matrix((g,(g_row, g_col)))
	print(stuff)
	print(stuff.toarray())

	# resistor_values, nodes_resistors, nodes_voltages, voltage_values, nodes_inductors
	poopoothing = flatten([nodes_resistors, nodes_voltages, nodes_inductors])
	print("bitch: "+str(poopoothing))
	print("[nodes_resistors, nodes_voltages, nodes_inductors] == "+str([nodes_resistors, nodes_voltages, nodes_inductors]))
	poopoothing = max(poopoothing)

	print("Resulting matrix: "+str(resulting_matrix))
	return resulting_matrix, poopoothing
	#return resulting_matrix, max_shit

import sys

def parse_file():
	# resistor_nodes, resistor_values = parse_file()
	if len(sys.argv) != 2: # File input?
		print("Usage: "+str(sys.argv[0])+" INPUT_NET_FILE")
		exit(0)

	resistor_nodes = []
	resistor_values = []
	inductor_values = []
	nodes_inductors = []
	nodes_capacitors = []
	capacitor_values = []

	diode_nodes = []  # [[n+, n-], ...]
	Is_values = []    # [1e-12, ...]

	fh = open(sys.argv[-1], "r")
	lines = fh.readlines()
	fh.close()
	voltage_nodes, voltage_values = [], []
	for line in lines:
		
		if "\n" == line[-1]:
			line = line[:-1]
		print("Line: "+str(line))
		if line == "":
			continue
		if "R" == line[0]:
			
			# resistor

			tokens = line.split(" ")
			tokens = [thing for thing in tokens if thing != ""]
			print("tokens: "+str(tokens))
			index_thing = [int(x) for x in tokens[1:3]]
			value = float(tokens[-1])

			resistor_nodes.append(index_thing)
			resistor_values.append(value)

		if "V" == line[0]:
			
			# resistor

			tokens = line.split(" ")
			tokens = [thing for thing in tokens if thing != ""]
			print("tokens: "+str(tokens))
			index_thing = [int(x) for x in tokens[1:3]]
			value = float(tokens[-1])

			voltage_nodes.append(index_thing)
			voltage_values.append(value)
		
		if "L" == line[0]:

			# inductor:

			tokens = line.split(" ")
			tokens = [thing for thing in tokens if thing != ""]
			print("tokens: "+str(tokens))
			index_thing = [int(x) for x in tokens[1:3]]
			value = float(tokens[-1])

			nodes_inductors.append(index_thing)
			inductor_values.append(value)
			#nodes_inductors

		if "C" == line[0]:

			tokens = line.split(" ")
			tokens = [thing for thing in tokens if thing != ""]

			index_thing = [int(x) for x in tokens[1:3]]
			value = float(tokens[-1])

			nodes_capacitors.append(index_thing)
			capacitor_values.append(value)

		if "D" == line[0]:
			tokens = line.split(" ")
			tokens = [thing for thing in tokens if thing != ""]

			index_thing = [int(x) for x in tokens[1:3]]
			value = float(tokens[-1])

			diode_nodes.append(index_thing)
			Is_values.append(value)



	return resistor_nodes, resistor_values, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values, diode_nodes, Is_values


def generate_rhs(stuff, voltages):
	#print("resistances: "+str(resistances))
	print("voltages: "+str(voltages))
	return [0]*(stuff-len(voltages))+voltages



def get_initial_stuff(resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values):

	poopooshit = copy.deepcopy(voltage_values)
	poopooshit2 = copy.deepcopy(voltage_nodes)

	for val_index, cap_node in enumerate(nodes_capacitors):

		poopooshit.append(capacitor_values[val_index])
		poopooshit2.append(cap_node)

	return resistor_values, resistor_nodes, poopooshit2, poopooshit, nodes_inductors, inductor_values, [], []


# Newton rhapson for non-linear components...

def newton_raphson_solve(G, rhs, diode_nodes, Is_values, V_guess, max_iters=50):
    VT = 0.025

    V = np.asarray(V_guess).reshape(-1, 1)  # FORCE (N,1)

    for _ in range(max_iters):
        Gnl = G.copy()
        rhs_nl = rhs.copy()

        for i, diode in enumerate(diode_nodes):
            n_plus, n_minus = diode

            Vp = V[n_plus - 1, 0] if n_plus != 0 else 0.0
            Vm = V[n_minus - 1, 0] if n_minus != 0 else 0.0
            Vd = Vp - Vm

            Is = Is_values[i]
            expV = np.exp(Vd / VT)

            I = Is * (expV - 1.0)
            Gd = Is / VT * expV
            Ieq = I - Gd * Vd

            if n_plus != 0:
                Gnl[n_plus-1, n_plus-1] += Gd
                rhs_nl[n_plus-1, 0] -= Ieq
            if n_minus != 0:
                Gnl[n_minus-1, n_minus-1] += Gd
                rhs_nl[n_minus-1, 0] += Ieq
            if n_plus != 0 and n_minus != 0:
                Gnl[n_plus-1, n_minus-1] -= Gd
                Gnl[n_minus-1, n_plus-1] -= Gd

        deltaV = np.linalg.solve(Gnl, rhs_nl)
        V += deltaV

        if np.max(np.abs(deltaV)) < 1e-9:
            break

    return V  # ALWAYS (N,1)



if __name__=="__main__":

	#resistor_node_num = [int(x) for x in input("Enter resistor nodes: ").split(" ")]

	file = True
	if not file:

		resistor_node_num = int(input("How many resistor node pairs? : "))
		
		resistor_nodes = []

		resistor_values = []

		while len(resistor_nodes) < resistor_node_num:

			appended_thing = [int(x) for x in input("Enter node pair: ").split(" ")]
			if len(appended_thing) == 2:

				resistor_nodes.append(appended_thing)
				resistor_value = float(input("Give the resistor value: "))
				resistor_values.append(resistor_value)
			else:
				print("Please input two integer numbers separated by a space.")
	else:
		# return resistor_nodes, resistor_values, voltage_nodes, voltage_values, nodes_inductors, inductor_values
		resistor_nodes, resistor_values, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values, diode_nodes, Is_values = parse_file()


	poopoofirst = [resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values]
	poopoofirst = [copy.deepcopy(x) for x in poopoofirst]


	resistor_values2, resistor_nodes2, voltage_nodes2, voltage_values2, nodes_inductors2, inductor_values2, nodes_capacitors2, capacitor_values2 = get_initial_stuff(resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values)
	print("voltage_nodes after: "+str(voltage_nodes))
	
	if poopoofirst != [resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values]:
		print("Fail")
		exit(1)
	G, max_node_num = get_conductance_matrix(resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors)

	Ginitial, max_node_num2 = get_conductance_matrix(resistor_values2, resistor_nodes2, voltage_nodes2, voltage_values2, nodes_inductors2)

	print("G == "+str(G))
	
	print("[len(G[0]), len(G)] == "+str([len(G[0]), len(G)]))


	C, _ = get_capacitance_and_inductance_matrix(nodes_inductors, inductor_values, nodes_capacitors, capacitor_values, max_node_num, [len(G[0]), len(G)])


	
	print(resistor_nodes)
	#stuff = len(Ginitial[0])

	stuff = len(G[0])
	#rhs = generate_rhs(stuff, voltage_values2)

	rhs = generate_rhs(stuff, voltage_values)


	stuff2 = len(Ginitial[0])
	rhs2 = generate_rhs(stuff2, voltage_values2)

	x = np.linalg.lstsq(np.array(Ginitial), np.array(rhs2))
	x = x[0]

	# solution = np.concatenate((x[:max_node_num2],np.array(inductor_values),x[max_node_num2:(max_node_num2+len(voltage_nodes))]))
	
	solution = x.reshape((-1, 1))


	G = np.array(G)
	C = np.array(C)
	rhs = np.array(rhs)


	print("Result: "+str(G))
	print("Solution: "+str(solution))
	h = 0.01
	count = 1000
	print()
	K = C + 0.5*h*G

	# solution = np.array([solution])
	# solution = solution.T



	print("K == "+str(K))
	print("solutionpoopoo == "+str(solution))
	print("Initial condition is this: "+str(solution))

	solutions = [solution]

	# original_rhs = copy.deepcopy(np.array([rhs]).T)

	original_rhs = np.asarray(rhs).reshape(-1, 1)


	print("original_rhs == "+str(original_rhs))
	
	x_vals = [0]
	cur_x = 0

	for _ in range(count-1):
		# rhs = (net.C - 0.5 * h * net.G) * net.x[:, k - 1] + 0.5 * h * (rhs_fun(net.t[k - 1]) + rhs_fun(net.t[k]))
		#rhs = (C - 0.5 * h * G) * solution + 0.5 * h * (original_rhs * 2)
		# poopoo = np.dot((C - 0.5 * h * G),solution)
		rhs = np.dot((C - 0.5 * h * G),solution) + 0.5 * h * (original_rhs * 2)
		#net.x[:, k] = spsolve(K, rhs)
		# solution = np.linalg.solve(K, rhs)
		solution = newton_raphson_solve(G, rhs, diode_nodes, Is_values, V_guess)
		solutions.append(solution)
		x_vals.append(cur_x)
		cur_x += h

	print("="*100)
	print("Final sols:")
	print(solutions)

	print("="*100)

	print(solutions[0])
	print(solutions[1])

	# plot solution
	
	print("solutions[0] == "+str(solutions[0]))

	y_vals = [sol[2][0] for sol in solutions]

	plt.plot(x_vals, y_vals)
	plt.show()
	exit(0)
