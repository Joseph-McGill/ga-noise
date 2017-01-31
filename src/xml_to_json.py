import xml.etree.ElementTree
import json
import sys

## Joseph McGill
## Fall 2016
## This program converts travelling salesman problem instnaces from xml
## to json for use by the main program.

# exit if no files are passed in
if len(sys.argv) < 2:
    print("No arguments given")

else:

    # parse each file passed in as command line arguments
    for file_name in sys.argv[1:]:

        # append the file paths
        infile = '../data/tsp_xml/' + file_name + '.xml'
        outfile = '../data/' + file_name + '.json'

        # optimal distances for each supported TSP instance
        opt_distances = {'burma14': 3323, 'bays29': 2020, 'dantzig42': 699,
                        'eil51': 426, 'ulysses16': 6859, 'ulysses22': 7013,
                        'att48': 10628, 'eil76': 538}

        # parse the xml files
        tree = xml.etree.ElementTree.parse(infile)
        graph = tree.getroot().find('graph')

        # get the distance costs for each node in the TSP instance
        vertices = []
        for i, a in enumerate(graph.findall('vertex')):
            edges = []
            for b in a.findall('edge'):
                edges.append(float(b.get('cost')))
            edges.insert(i, 0)
            vertices.append(edges)

        # put the data into a dictionary
        data = {}
        data['TourSize'] = len(vertices)
        data['OptDistance'] = opt_distances[file_name]
        data['DistanceMatrix'] = vertices

        # output the data to the json files
        with open(outfile, 'w') as out:
            json.dump(data, out, sort_keys=False, indent=4,
                      separators=(',', ': '))
