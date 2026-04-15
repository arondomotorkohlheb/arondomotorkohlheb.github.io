# this script establishes the controller that will be used to control the heating/cooling modes

class Node():
    def __init__(self, id, name, output) -> None:
        self.id = id
        self.name = name
        self.output = output
        self.input_conditions = []
        self.links = []
    
    def assign_link(self, link):
        self.links.append(link)

    def assign_input_condition(self, condition):
        self.input_conditions.append(condition)

# the nodes of the graph
nodes_dict = {
    'heating, window closed': (0,0,0,0,1),
    'cooling, window closed': (0,0,0,0,1),
    'turning valves 1 and 2 +': (0,1,1,0,0),
    'turning valves 1 and 2 -': (0,-1,-1,0,0),
    'none, window closed': (0,0,0,0,0),
    'turning valve 0 +': (1,0,0,0,0),
    'closing window': (0,0,0,0,0),
    'opening window': (0,0,0,1,0),
    'turning valve 0 -': (-1,0,0,0,0),
    'none, window opened': (0,0,0,0,0),
}
nodes = {}
i = 1
for node in nodes_dict:
    nodes[i] = Node(i, node, nodes_dict[node])
    i += 1

class Link():
    def __init__(self,node_from, node_to, condition) -> None:
        self.node_from = node_from # node objects
        self.node_to = node_to
        self.condition = condition
    
    def custom_print(self):
        print(self.node_from.name, '|', self.node_to.name,  '|',self.condition)

# for each link: id of node from, id of node to, condition (required input signal to traverse along the link)
# this is the same as the handed in reduced graph, just written down in this format
# the id-s of the nodes are indicated on the top right corner of each node
linkes_lst = [
    ((1,4), (1,0,1,1,0)),
    ((2,3), (2,0,-1,-1,0)),
    ((2,3), (1,0,-1,-1,0)),
    ((2,3), (0,0,-1,-1,0)),
    ((3,1) , (0,0,1,1,0)),
    ((3,4) , (3,0,0,0,0)),
    ((3,4) , (1,0,-1,-1,0)),
    ((3,5) , (1,0,0,0,0)),
    ((3,6) , (2,0,0,0,0)),
    ((4,2) , (3,0,-1,-1,0)),
    ((4,3) , (0,0,-1,-1,0)),
    ((4,3) , (2,0,-1,-1,0)),
    ((4,5) , (1,0,0,0,0)),
    ((5,3) , (0,0,0,0,0)),
    ((5,3) , (3,0,0,0,0)),
    ((5,6) , (2,0,0,0,0)),
    ((6,4) , (3,0,0,0,0)),
    ((6,5) , (0,0,0,0,0)),
    ((6,5) , (1,0,0,0,0)),
    ((6,8) , (2,1,0,0,0)),
    ((7,6) , (0,-1,0,0,0)),
    ((7,6) , (1,-1,0,0,0)),
    ((7,6) , (2,-1,0,0,0)),
    ((7,6) , (3,-1,0,0,0)),
    ((8,9) , (0,1,0,0,2)),
    ((8,9) , (1,1,0,0,2)),
    ((8,9) , (2,1,0,0,2)),
    ((8,9) , (3,1,0,0,2)),
    ((9,7) , (0,-1,0,0,2)),
    ((9,7) , (1,-1,0,0,2)),
    ((9,7) , (3,-1,0,0,2)),
    ((9,10) , (2,0,0,0,2)),
    ((10,9) , (0,0,0,0,2)),
    ((10,9) , (1,0,0,0,2)),
    ((10,9) , (3,0,0,0,2)),
]

links = []
for link in linkes_lst:
    new_link = Link(nodes[link[0][0]], nodes[link[0][1]], link[1])
    links.append(new_link)
    new_link.node_from.assign_link(new_link)
    new_link.node_to.assign_input_condition(new_link.condition)


class Controller():
    def __init__(self, initial_Tr, nodes = nodes, links = links) -> None:
        # assigning starting point
        self.nodes = nodes
        self.links = links
        if initial_Tr == 0:
            self.node = nodes[1]
        elif initial_Tr == 1:
            self.node = nodes[5]
        elif initial_Tr == 2:
            self.node = nodes[10]
        elif initial_Tr == 3:
            self.node = nodes[2]
        else:
            raise ValueError
        self.previous_node = self.node

    @property
    def output(self):
        return self.node.output
    
    def update(self, input):
        for link in self.node.links:
            if input == link.condition:
                self.previous_node = self.node
                self.node = link.node_to
                return 1
        return 0

    def health_check(self, input):
        for cond in self.node.input_conditions:
            if input == cond:
                return 1
        return 0

    
    

# c = Controller(2)

# [link.custom_print() for link in c.node.links]



#if __name__ == "__main__":
  
  #  main()