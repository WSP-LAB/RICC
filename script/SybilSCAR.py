# This code was written by Dongwon Shin. 
# It implements SybilSCAR algorithm proposed in the following paper:
# Binghui Wang et al. "SybilSCAR: Sybil Detection in Online Social Networks via Local Rule based Propagation", INFOCOM, 2017. 
#
# For the official implementation, check the following link:
# https://github.com/binghuiwang/sybildetection

import constants
import copy


def read_graph(graph, _type, attacked, attack, option=""):
    """ read the graph file and save edge information to 'constants.graph_list'
    Because we deal with undirected graph, each edge must appear twice
    :return:
    """
    if attacked:
        f_graph = open("../dataset/graph/{}/newgraph_{}_{}_{}{}.txt".format(graph, graph, _type, attack, option), "r")
        print("Opening: newgraph_{}_{}_{}{}.txt".format(graph, _type, attack, option))
    else:
        f_graph = open("../dataset/graph/{}/originalgraph_{}.txt".format(graph, graph), "r")

    lines = f_graph.read().splitlines()

    for line in lines:
        line = line.split()
        node1 = int(line[0])
        node2 = int(line[1])

        # add edge to 'graph_list'
        if attacked:
            constants.graph_list_attacked[node1].append(node2)
        else:
            constants.graph_list_original[node1].append(node2)

    # number of nodes in the graph
    constants.node_num = len(constants.graph_list_original)
    
    f_graph.close()


def read_prior(graph, prior_path, prior_list, theta, is_train):
    """ assig`n `prior scores.
    If you have the training dataset,
    this function automatically converts the training dataset into prior scores and
    assign prior score to 'constants.prior_list'
    Or if you have the prior socre file, this function assign prior score to 'constants.prior_list'
    :return:
    """

    # If you have training dataset
    if is_train:
        f_train = open("../dataset/graph/{}/train.txt".format(graph), "r")

        train_negative = f_train.readline()
        train_positive = f_train.readline()

        negative_nodes = train_negative.split()
        positive_nodes = train_positive.split()

        for negative_idx in negative_nodes:
            prior_list[int(negative_idx)] = -1 * theta
        for positive_idx in positive_nodes:
            prior_list[int(positive_idx)] = +1 * theta

        f_train.close()

    # If you have prior file
    else:
        f_prior = open(prior_path, "r")

        lines = f_prior.read().splitlines()

        for line in lines:
            line = line.split()

            node_idx = int(line[0])
            prior_score = float(line[1])

            prior_list[node_idx] = prior_score

        f_prior.close()

    return prior_list


def run_lbp(prior_list, post_list, wei, attacked):
    """ run LinLBP one iteration with 'graph_list' & 'prior_list'
    :return:
    """
    if attacked:
        graph_list = constants.graph_list_attacked
    else:
        graph_list = constants.graph_list_original
        
    next_post_list = [0.0] * constants.n

    for nei_list in enumerate(graph_list):
        #score_tmp = float(post_list[nei_list[0]])
        score_tmp = 0
        prior_score = prior_list[nei_list[0]]

        for nei in nei_list[1]:
            post_score = post_list[int(nei)]
            score_tmp += (2.0 * wei * post_score)

        score_tmp += prior_score

        if score_tmp > 0.5:
            score_tmp = 0.5
        if score_tmp < -0.5:
            score_tmp = -0.5

        next_post_list[nei_list[0]] = score_tmp

    post_list = copy.deepcopy(next_post_list)

    return post_list


def save_posterior(post_path, post_list):
    """ save posterior scores in file.
    :return:
    """
    f_post = open(post_path, "w")

    for line in enumerate(post_list):
        f_post.write('{} {:.10f}\n'.format(line[0], line[1]))
        
    f_post.close()


def init(graph, prior_path, post_path, iteration, theta, wei, is_train, attacked):
    prior_list = [0] * constants.n

    prior_list = read_prior(graph, prior_path, prior_list, theta, is_train)

    post_list = copy.deepcopy(prior_list)

    for _ in range(iteration):
        post_list_tmp = run_lbp(prior_list, post_list, wei, attacked)
        post_list = copy.deepcopy(post_list_tmp)

    save_posterior(post_path, post_list)
    
