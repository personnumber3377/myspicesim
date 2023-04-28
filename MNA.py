
from scipy.sparse import csr_matrix
import numpy as np
import copy
import matplotlib.pyplot as plt



def generate_matrix(indexes, values, shape=None):
	print("Indexes is this: "+str(indexes))
	print("Values is this: "+str(values))
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
	print("All nodes: "+str(all_nodes))
	return max(flatten(all_nodes))



# C = get_capacitance_and_inductance_matrix(nodes_inductors, inductor_values, nodes_capacitors, capacitor_values)

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
	#return resulting_matrix, max_shit
	'''
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
	'''



def get_conductance_matrix(resistor_values, nodes_resistors, nodes_voltages, voltage_values, nodes_inductors):
	

	max_nodes = get_max_node_num(nodes_resistors+nodes_voltages+nodes_inductors)

	'''
	        for ir in indexR:
            # get nores
            print("Resistor")
            print("self.nodes: "+str(self.nodes))
            N1, N2 = self.nodes[ir]

            # detect connection
            if (N1 == 0) or (N2 == 0): # if grounded...
                # diagonal term
                print("Grounded")
                g.append(1.0 / self.values[ir])
                g_row.append(max([N1, N2]) - 1)
                g_col.append(max([N1, N2]) - 1)

            else:                      # if not grounded...
                # diagonal term
                g.append(1.0 / self.values[ir])
                g_row.append(N1 - 1)
                g_col.append(N1 - 1)

                # diagonal term
                g.append(1.0 / self.values[ir])
                g_row.append(N2 - 1)
                g_col.append(N2 - 1)

                # N1-N2 term
                g.append(-1.0 / self.values[ir])
                g_row.append(N1 - 1)
                g_col.append(N2 - 1)

                # N2-N1 term
                g.append(-1.0 / self.values[ir])
                g_row.append(N2 - 1)
                g_col.append(N1 - 1)
	'''

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

	'''
        for k, il in enumerate(indexL):
            # get nodes
            print("Inductor")
            N1, N2 = self.nodes[il]

            # detect connection
            if N1 == 0:  # if grounded to N1 ...
                # negative terminal
                g.append(-1)
                g_row.append(N2 - 1)
                g_col.append(self.node_num + k)

                # negative terminal
                g.append(-1)
                g_row.append(self.node_num + k)
                g_col.append(N2 - 1)

            elif N2 == 0:  # if grounded to N2 ...
                # positive terminal
                g.append(1)
                g_row.append(N1 - 1)
                g_col.append(self.node_num + k)

                # positive terminal
                g.append(1)
                g_row.append(self.node_num + k)
                g_col.append(N1 - 1)

            else:  # if not grounded ...
                # positive terminal
                print("poopooshitoof")
                print("N1 == "+str(N1))
                print("N2 == "+str(N2))
                g.append(1)
                g_row.append(N1 - 1)
                g_col.append(self.node_num + k)

                # positive terminal
                g.append(1)
                g_row.append(self.node_num + k)
                g_col.append(N1 - 1)

                #print("")

                # negative terminal
                g.append(-1)
                g_row.append(N2 - 1)
                g_col.append(self.node_num + k)

                # negative terminal
                g.append(-1)
                g_row.append(self.node_num + k)
                g_col.append(N2 - 1)

	'''
	index_stuff = [[g_row[i],g_col[i]] for i in range(len(g_col))]
	'''
	if len(g_col) != 0 and len(g_row) != 0:


		resulting_matrix = generate_matrix(index_stuff, g)
		max_nodes = len(resulting_matrix[0])
		print("Matrix before voltage sources: "+str(resulting_matrix))
	else:
		max_nodes = 0
	'''
	#max_nodes = max(g_row)


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





	'''
	Expected values:

	g == [1.0, 1.0, -1.0, -1.0, 0.5, 0.5, -0.5, -0.5, 0.1, 0.1, -0.1, -0.1, 0.1]
	g_row == [1, 0, 1, 0, 1, 0, 1, 0, 2, 1, 2, 1, 2]
	g_col == [1, 0, 0, 1, 1, 0, 0, 1, 2, 1, 1, 2, 2]

	'''

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

	resistor_nodes = []
	resistor_values = []
	inductor_values = []
	nodes_inductors = []
	nodes_capacitors = []
	capacitor_values = []

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




	return resistor_nodes, resistor_values, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values


def generate_rhs(stuff, voltages):
	#print("resistances: "+str(resistances))
	print("voltages: "+str(voltages))
	return [0]*(stuff-len(voltages))+voltages



def get_initial_stuff(resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values):

	'''
	for iv in indexV:
        Vnum = int(net.names[iv][1:])

        if isinstance(net.values[iv],list):
            tsr_fun = getattr(tsr, net.source_type[net.names[iv]])
            net_op.values[iv] = tsr_fun(*net.values[iv], t=0)

        if Vnum >= nv:
            nv = Vnum + 1
    ni = 1  # get max IDnumber for current sources
    for ii in indexI:
        Inum = int(net.names[ii][1:])

        if isinstance(net.values[ii],list):
            tsr_fun = getattr(tsr, net.source_type[net.names[ii]])
            net_op.values[ii] = tsr_fun(*net.values[ii], t=0)

        if Inum >= ni:
            ni = Inum + 1

    # transform inductors (to current sources)
    for k, il in enumerate(indexL):
        new_name = 'I' + str(ni + k)
        track_change[new_name] = net_op.names[il]
        net_op.values[il] = net_op.IC[net_op.names[il]]
        net_op.names[il] = new_name

    # transform capacitors (to voltage sources)
    for k, ic in enumerate(indexC):
        new_name = 'V' + str(nv + k)
        track_change[new_name] = net_op.names[ic]
        net_op.values[ic] = net_op.IC[net_op.names[ic]]
        net_op.names[ic] = new_name

	'''

	#nv = max(flatten(voltage_nodes))

	#resistor_values_copy, resistor_nodes_copy, voltage_nodes_copy, voltage_values_copy, nodes_inductors_copy, inductor_values_copy, nodes_capacitors_copy, capacitor_values_copy = 
	poopooshit = copy.deepcopy(voltage_values)
	poopooshit2 = copy.deepcopy(voltage_nodes)

	for val_index, cap_node in enumerate(nodes_capacitors):

		poopooshit.append(capacitor_values[val_index])
		poopooshit2.append(cap_node)

	return resistor_values, resistor_nodes, poopooshit2, poopooshit, nodes_inductors, inductor_values, [], []









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
		resistor_nodes, resistor_values, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values = parse_file()

	print(resistor_nodes)


	print("voltage_nodes before: "+str(voltage_nodes))

	poopoofirst = [resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values]
	poopoofirst = [copy.deepcopy(x) for x in poopoofirst]


	resistor_values2, resistor_nodes2, voltage_nodes2, voltage_values2, nodes_inductors2, inductor_values2, nodes_capacitors2, capacitor_values2 = get_initial_stuff(resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values)
	print("voltage_nodes after: "+str(voltage_nodes))
	
	if poopoofirst != [resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values]:
		print("Fail")
		exit(1)
	G, max_node_num = get_conductance_matrix(resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors)
	
	print("[resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values] == "+str([resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors, inductor_values, nodes_capacitors, capacitor_values]))

	print("[resistor_values2, resistor_nodes2, voltage_nodes2, voltage_values2, nodes_inductors2] == "+str([resistor_values2, resistor_nodes2, voltage_nodes2, voltage_values2, nodes_inductors2]))

	print("Getting initial shit:")
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

	print("rhs: "+str(rhs))
	print("result: "+str(G))
	print("C == "+str(C))
	print("Ginitial == "+str(Ginitial))
	print("rhs2 == "+str(rhs2))
	#print("Ginitial == "+str(Ginitial))

	#solution = np.linalg.solve(np.array(Ginitial), np.array(rhs))

	#x = np.linalg.solve(np.array(Ginitial), np.array(rhs2))
	# lstsq

	x = np.linalg.lstsq(np.array(Ginitial), np.array(rhs2))
	x = x[0]
	print("x initial: "+str(x))
	print("x[0] == "+str(x[0]))
	print("max_node_num2 == "+str(max_node_num2))
	print("x[:max_node_num2] == "+str(x[:max_node_num2]))
	print("max_node_num2 == "+str(max_node_num2))
	print("inductor_values == "+str(inductor_values))
	print("x[max_node_num2:(max_node_num2+len(voltage_nodes))] == "+str(x[max_node_num2:(max_node_num2+len(voltage_nodes))]))
	solution = np.concatenate((x[:max_node_num2],np.array(inductor_values),x[max_node_num2:(max_node_num2+len(voltage_nodes))]))

	

	print("Initial condition solution: "+str(solution))


	# net_op.node_num is basically this max_node_num2

	# At this point the solution is the initial conditions basically.

	'''
	net.x[:, 0] = np.concatenate((net_op.x[:net_op.node_num],
                                  np.array(net_op.values)[sorted(net.isort[1])],
                                  net_op.x[net_op.node_num:(net_op.node_num + NV)],
                                  net_op.x[(net_op.node_num + NV):(net_op.node_num + NV + NE)],
                                  net_op.x[(net_op.node_num + NV + NE):(net_op.node_num + NV + NE + NH)]))

    # Solution (Integration using trepezoidal rule. Ref: Vlach, eq 9.4.6, pag. 277)
    K = net.C + 0.5 * h * net.G
    for k in range(1, net.t.size):
        rhs = (net.C - 0.5 * h * net.G) * net.x[:, k - 1] + 0.5 * h * (rhs_fun(net.t[k - 1]) + rhs_fun(net.t[k]))

        net.x[:, k] = spsolve(K, rhs)
        print(net.x[:, k])
        print("rhs == "+str(rhs))
        print("K == "+str(K.toarray()))
	'''


	'''

	    net_op = deepcopy(net)

    # dictionary to track changes (L <-> I and C <-> V)
    track_change = {}

    # reorder original network and get index
    net_op.reorder()
    indexL = sorted(net_op.isort[1])
    print("Poopooshit oof: "+str(indexL))
    indexC = sorted(net_op.isort[2])
    indexV = sorted(net_op.isort[3])
    indexI = sorted(net_op.isort[4])

    nv = 1  # get max IDnumber for voltage sources
    print("net_op values before: "+str(net_op.values))

    for iv in indexV:
        Vnum = int(net.names[iv][1:])

        if isinstance(net.values[iv],list):
            tsr_fun = getattr(tsr, net.source_type[net.names[iv]])
            net_op.values[iv] = tsr_fun(*net.values[iv], t=0)

        if Vnum >= nv:
            nv = Vnum + 1
    ni = 1  # get max IDnumber for current sources
    for ii in indexI:
        Inum = int(net.names[ii][1:])

        if isinstance(net.values[ii],list):
            tsr_fun = getattr(tsr, net.source_type[net.names[ii]])
            net_op.values[ii] = tsr_fun(*net.values[ii], t=0)

        if Inum >= ni:
            ni = Inum + 1

    # transform inductors (to current sources)
    for k, il in enumerate(indexL):
        new_name = 'I' + str(ni + k)
        track_change[new_name] = net_op.names[il]
        net_op.values[il] = net_op.IC[net_op.names[il]]
        net_op.names[il] = new_name

    # transform capacitors (to voltage sources)
    for k, ic in enumerate(indexC):
        new_name = 'V' + str(nv + k)
        track_change[new_name] = net_op.names[ic]
        net_op.values[ic] = net_op.IC[net_op.names[ic]]
        net_op.names[ic] = new_name
	'''

	#net_initial = copy.deepcopy()

	# Get initial conditions by replacing the shit G, max_node_num = get_conductance_matrix(resistor_values, resistor_nodes, voltage_nodes, voltage_values, nodes_inductors)

	


	print("C before numpy: "+str(C))

	G = np.array(G)
	C = np.array(C)
	rhs = np.array(rhs)


	print("Result: "+str(G))
	print("Solution: "+str(solution))
	h = 0.01
	count = 1000
	print()
	K = C + 0.5*h*G

	solution = np.array([solution])
	solution = solution.T
	print("K == "+str(K))
	print("solutionpoopoo == "+str(solution))
	print("Initial condition is this: "+str(solution))

	####

	#solution = np.array([[ 5.,   0.,  -0.5]]).T

	####


	solutions = [solution]

	original_rhs = copy.deepcopy(np.array([rhs]).T)
	print("original_rhs == "+str(original_rhs))
	
	x_vals = [0]
	cur_x = 0

	for _ in range(count-1):
		# rhs = (net.C - 0.5 * h * net.G) * net.x[:, k - 1] + 0.5 * h * (rhs_fun(net.t[k - 1]) + rhs_fun(net.t[k]))
		#rhs = (C - 0.5 * h * G) * solution + 0.5 * h * (original_rhs * 2)
		rhs = np.dot((C - 0.5 * h * G),solution) + 0.5 * h * (original_rhs * 2)
		print("="*30)
		print("C == "+str(C))
		print("G == "+str(G))
		print("solution == "+str(solution))
		print("(C - 0.5 * h * G) * solution == "+str((C - 0.5 * h * G) * solution))
		print("np.dot((C - 0.5 * h * G),solution) == "+str(np.dot((C - 0.5 * h * G),solution)))
		print("0.5 * h * (original_rhs * 2) == "+str(0.5 * h * (original_rhs * 2)))
		print("K == "+str(K))
		print("rhs == "+str(rhs))
		print("="*30)
		#net.x[:, k] = spsolve(K, rhs)
		solution = np.linalg.solve(K, rhs)
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


	right_results = [5.00000000e-01, 4.52380952e-01, 4.09297052e-01, 3.70316381e-01
, 3.35048154e-01, 3.03138806e-01, 2.74268443e-01, 2.48147639e-01
, 2.24514531e-01, 2.03132194e-01, 1.83786271e-01, 1.66282817e-01
, 1.50446358e-01, 1.36118133e-01, 1.23154502e-01, 1.11425502e-01
, 1.00813549e-01, 9.12122586e-02, 8.25253769e-02, 7.46658172e-02
, 6.75547870e-02, 6.11209977e-02, 5.52999503e-02, 5.00332884e-02
, 4.52682133e-02, 4.09569549e-02, 3.70562925e-02, 3.35271218e-02
, 3.03340626e-02, 2.74451042e-02, 2.48312848e-02, 2.24664005e-02
, 2.03267433e-02, 1.83908630e-02, 1.66393523e-02, 1.50546520e-02
, 1.36208757e-02, 1.23236494e-02, 1.11499685e-02, 1.00880667e-02
, 9.12729848e-03, 8.25803196e-03, 7.47155273e-03, 6.75997628e-03
, 6.11616901e-03, 5.53367672e-03, 5.00665989e-03, 4.52983514e-03
, 4.09842227e-03, 3.70809634e-03, 3.35494431e-03, 3.03542580e-03
, 2.74633763e-03, 2.48478167e-03, 2.24813579e-03, 2.03402762e-03
, 1.84031071e-03, 1.66504302e-03, 1.50646749e-03, 1.36299440e-03
, 1.23318541e-03, 1.11573918e-03, 1.00947831e-03, 9.13337514e-04
, 8.26352989e-04, 7.47652704e-04, 6.76447685e-04, 6.12024096e-04
, 5.53736087e-04, 5.00999317e-04, 4.53285096e-04, 4.10115087e-04
, 3.71056507e-04, 3.35717792e-04, 3.03744669e-04, 2.74816605e-04
, 2.48643595e-04, 2.24963253e-04, 2.03538181e-04, 1.84153593e-04
, 1.66615155e-04, 1.50747045e-04, 1.36390184e-04, 1.23400642e-04
, 1.11648200e-04, 1.01015038e-04, 9.13945585e-05, 8.26903148e-05
, 7.48150467e-05, 6.76898042e-05, 6.12431562e-05, 5.54104746e-05
, 5.01332866e-05, 4.53586879e-05, 4.10388128e-05, 3.71303545e-05
, 3.35941302e-05, 3.03946892e-05, 2.74999569e-05, 2.48809134e-05]

	print("Errors: ")
	print([(-1*y_vals[i])- right_results[i] for i in range(len(y_vals))])


	exit(0)


'''

[[1.5, -1.5, 0.0, 0.0],
[-1.5, 1.6, -0.1, 0.0],
[0.0, -0.1, 0.2, -0.1],
[0.0, 0.0, -0.1, 0.2]]


[[1.5, -1.5, 0.0, 0.0],
[-1.5, 1.6, -0.1, 0.0],
[0.0, -0.1, 0.2, -0.1],
[0.0, 0.0, -0.1, 0.30000000000000004]]

[[1.5, -1.5, 0.0, 0.0],
[-1.5, 1.6, -0.1, 0.0],
[0.0, -0.1, 0.30000000000000004, -0.2], 
[0.0, 0.0, -0.2, 0.4]]

[[1.5, -1.5, 0.0, 0.0],
[-1.5, 1.6, -0.1, 0.0],
[0.0, -0.1, 0.4, -0.2],
[0.0, 0.0, -0.2, 0.4]]

[[1.5, -1.5, 0.0, 0.0],
[-1.5, 1.7000000000000002, -0.1, -0.1],
[0.0, -0.1, 0.30000000000000004, -0.2],
[0.0, -0.1, -0.2, 0.5]]
'''