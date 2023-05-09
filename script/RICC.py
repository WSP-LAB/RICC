import os.path
import random
import math
import yaml
from sklearn.metrics import roc_auc_score
import argparse

import constants
import SybilSCAR
import utils


def choose_random_train_set():
    """ choose random training dataset according to the former posterior score
    :return:
    """
    # call global variables
    turn = constants.turn

    # For new trainset
    train_negative = []
    train_positive = []

    # read former posterior score
    negative_list = []
    positive_list = []

    f_score = open(score_path + "/{}/{}/posterior/{}/post_sybilscar_{}_evaluation.txt".format(graph, types, case, turn), "r")
    lines = f_score.read().splitlines()
    for line in lines:
        line = line.split()
        if float(line[1]) < threshold:
            negative_list.append(line[0])
        else:
            positive_list.append(line[0])
    f_score.close()

    while True:
        rand = random.choice(negative_list)
        if rand not in train_negative:
            train_negative.append(rand)

        if len(train_negative) == sampling_size:
            break

    while True:
        rand = random.choice(positive_list)
        if rand not in train_positive:
            train_positive.append(rand)

        if len(train_positive) == sampling_size:
            break

    # save training dataset as file
    f_train = open(result_path + "/trainset/{}/{}/{}/train_{}_prime.txt".format(graph, types, case, turn), "w")

    for i in range(sampling_size):
        f_train.write(str(train_negative[i]) + " ")
    f_train.write("\n")
    for i in range(sampling_size):
        f_train.write(str(train_positive[i]) + " ")
    f_train.write("\n")
    f_train.close()


def trainset2prior():
    """ change training dataset to prior score
    we assign +[theata] prior score to positive nodes and -[theta] prior score to negative nodes
    :return:
    """
    # call global variables
    turn = constants.turn
    num_negative = constants.num_negative
    num_positive = constants.num_positive
    num_unlabel = constants.num_unlabel

    f_train = open(result_path + "/trainset/{}/{}/{}/train_{}_prime.txt".format(graph, types, case, turn), "r")
    f_train_score = open(score_path + "/{}/{}/prior/{}/prior_{}_prime.txt".format(graph, types, case, turn), "w")
    train = f_train.read().splitlines()

    negative_train_node = train[0].split(" ")
    positive_train_node = train[1].split(" ")

    for i in range(0, num_negative + num_positive + num_unlabel):
        if str(i) in negative_train_node:
            f_train_score.write(str(i) + " -{}\n".format(theta))
        elif str(i) in positive_train_node:
            f_train_score.write(str(i) + " {}\n".format(theta))
        else:
            f_train_score.write(str(i) + " 0\n")
    if attack == "NNI":
        for i in range(num_negative + num_positive + num_unlabel, num_negative + num_positive + num_unlabel + constants.num_new_nodes):
            f_train_score.write(str(i) + " 0.1\n")

    f_train.close()
    f_train_score.close()


def run_lbp(flag):
    """ With "prior score of N (prime)", run LBP and compute "post_sybilscar_N_prime".
    :param flag: Determine the prior score's type ("prior N prime" for TRUE, "prior N" for FALSE)
    :return:
    """
    # call global variables
    turn = constants.turn

    # Determine the prior score's type (prime or not)
    # Prime
    if flag:
        # for prime (small weight)
        prior_path = score_path + "/{}/{}/prior/{}/prior_{}_prime.txt".format(graph, types, case, turn)
        post_path = score_path + "/{}/{}/posterior/{}/post_sybilscar_{}_prime.txt".format(graph, types, case, turn)
        SybilSCAR.init(graph, prior_path, post_path, iteration, theta, weight * buffer, is_train=False, attacked=True)

    # Original (not prime)
    else:
        # for optimization (small weight)
        prior_path = score_path + "/{}/{}/prior/{}/prior_{}.txt".format(graph, types, case, turn + 1)
        post_path = score_path + "/{}/{}/posterior/{}/post_sybilscar_{}.txt".format(graph, types, case, turn + 1)
        SybilSCAR.init(graph, prior_path, post_path, iteration, theta, weight * buffer, is_train=False, attacked=True)

        # for evaluation (higher weight)
        prior_path = score_path + "/{}/{}/prior/{}/prior_{}_evaluation.txt".format(graph, types, case, turn + 1)
        post_path = score_path + "/{}/{}/posterior/{}/post_sybilscar_{}_evaluation.txt".format(graph, types, case, turn + 1)
        SybilSCAR.init(graph, prior_path, post_path, iteration, theta, weight, is_train=False, attacked=True)


def compute_diff():
    """ compute the score diff with post N & post N prime
    :return:
    """
    # call global variables
    turn = constants.turn
    num_negative = constants.num_negative
    num_positive = constants.num_positive
    num_unlabel = constants.num_unlabel

    # compute posterior score
    if attack == "NNI":
        score_list = [0] * (constants.num_negative + constants.num_positive + constants.num_unlabel + constants.num_new_nodes)
    else:
        score_list = [0] * (constants.num_negative + constants.num_positive + constants.num_unlabel)

    f_posterior = open(score_path + "/{}/{}/posterior/{}/"
                                    "post_sybilscar_{}_prime.txt".format(graph, types, case, turn))
    lines = f_posterior.read().splitlines()
    for line in lines:
        line = line.split(" ")
        index = int(line[0])
        score = float(line[1])
        score_list[index] += score
    f_posterior.close()

    # for diff
    f_train_ori = open(dataset_path + "/graph/{}/train.txt".format(graph), "r")
    f_train_new = open(result_path + "/trainset/{}/{}/{}/train_{}_prime.txt".format(graph, types, case, turn), "r")
    f_posterior_former = open(score_path + "/{}/{}/posterior/{}/"
                                           "post_sybilscar_{}.txt".format(graph, types, case, turn), "r")
    f_diff = open(score_path + "/{}/{}/diff/{}/diff_{}.txt".format(graph, types, case, turn), "w")

    # read original trainset
    train_ori = []
    lines = f_train_ori.read().splitlines()
    for line in lines:
        line = line.split()
        train_ori.extend(line)
    # read new trainset
    train_new = []
    lines = f_train_new.read().splitlines()
    for line in lines:
        line = line.split()
        train_new.extend(line)


    # compare diff
    lines1 = f_posterior_former.read().splitlines()

    score_diff_list = []
    for line1, line2 in zip(lines1, score_list):
        line1 = line1.split()
        score1 = float(line1[1])
        score2 = float(line2)
        diff = float(score2 - score1)

        if len(lines1) != len(score_list):
            raise IndexError

        elif line1[0] in train_ori or math.isclose(score1, 0.0):
            f_diff.write(str(line1[0]) + " " + str(0) + "\n")

        # elif line1[0] in train_new:
        #     f_diff.write(str(line1[0]) + " " + str(0) + "\n")

        else:
            f_diff.write(str(line1[0]) + " " + str(diff) + "\n")

    f_posterior_former.close()
    f_diff.close()
    f_train_ori.close()


def update_trainset():
    """ update N+1 th prior score according to posterior score difference
    :return:
    """
    # call global variables
    turn = constants.turn

    f_train_former_eval = open(score_path + "/{}/{}/prior/{}/prior_{}_evaluation.txt".format(graph, types, case, turn), "r")
    f_train_former_opt = open(score_path + "/{}/{}/prior/{}/prior_{}.txt".format(graph, types, case, turn), "r")
    f_diff = open(score_path + "/{}/{}/diff/{}/diff_{}.txt".format(graph, types, case, turn), "r")
    f_train_new_eval = open(score_path + "/{}/{}/prior/{}/prior_{}_evaluation.txt".format(graph, types, case, turn + 1), "w")
    f_train_new_opt = open(score_path + "/{}/{}/prior/{}/prior_{}.txt".format(graph, types, case, turn + 1), "w")

    lines1 = f_train_former_eval.read().splitlines()
    lines3 = f_train_former_opt.read().splitlines()
    lines2 = f_diff.read().splitlines()

    for line1, line2, line3 in zip(lines1, lines2, lines3):
        line1 = line1.split(" ")
        line2 = line2.split(" ")
        line3 = line3.split(" ")
        prior_former_eval = float(line1[1])
        prior_former_opt = float(line3[1])
        diff = float(line2[1])

        new = prior_former_eval + (diff * lr)
        new_opt = prior_former_opt + (diff * buffer * lr)

        if line1[0] != line2[0]:
            print(line1, line2)
            raise IndexError

        f_train_new_eval.write(str(line1[0]) + " " + str(new) + "\n")
        f_train_new_opt.write(str(line1[0]) + " " + str(new_opt) + "\n")

    f_train_former_eval.close()
    f_train_former_opt.close()
    f_diff.close()
    f_train_new_eval.close()
    f_train_new_opt.close()


def check_FN_nodes():
    """ Check how well this defense system detects False Negative nodes
    :return:
    """
    # call global variables
    turn = constants.turn
    num_negative = constants.num_negative
    num_positive = constants.num_positive
    num_unlabel = constants.num_unlabel

    target_types = types.split("_")[-1]

    f_performance = open(result_path + "/performance/{}/{}/{}/{}_summary.txt".format(graph, types, case, case), "a")
    f_score = open(score_path + "/{}/{}/posterior/{}/post_sybilscar_{}_evaluation.txt".format(graph, types, case, turn), "r")
    if "_T_" in options:
        f_target = open(dataset_path + "/graph/{}/target_{}{}.txt".format(graph, target_types, options), "r")
    else:
        f_target = open(dataset_path + "/graph/{}/target_{}.txt".format(graph, target_types), "r")

    # read target nodes
    target_detect = 0
    target_list = []
    targets = f_target.readline()
    targets = targets.split()
    target_num = len(targets)
    for target in targets:
        target_list.append(target)

    # read original trainset
    f_train_ori = open(dataset_path + "/graph/{}/train.txt".format(graph), "r")
    train_ori = []
    lines = f_train_ori.read().splitlines()
    for line in lines:
        line = line.split()
        train_ori.extend(line)

    # compute the posterior score after defense in this turn
    score_list = []
    score_list_no_train = []
    scores = f_score.read().splitlines()

    for (idx, score) in enumerate(scores):
        score = score.split()
        score_list.append(score[1])
        if str(idx) not in train_ori:
            score_list_no_train.append(float(score[1]))

    # find the FN nodes
    if turn == 0:
        f1 = open(dataset_path + "/initial_files/{}/{}/post_sybilscar_before_attack({}_{}).txt".format(graph, types, weight, iteration), "r")
        f2 = open(score_path + "/{}/{}/posterior/{}/post_sybilscar_0_evaluation.txt".format(graph, types, case), "r")

        lines1 = f1.read().splitlines()
        lines2 = f2.read().splitlines()

        for line1, line2 in zip(lines1, lines2):
            line1 = line1.split()
            line2 = line2.split()

            if float(line1[1]) > threshold > float(line2[1]):
                if int(line1[0]) >= num_negative:
                    constants.FN_nodes.append(line1[0])

        f1.close()
        f2.close()

    # compute the error rates (FN rate)
    error = 0
    for node in constants.FN_nodes:
        if float(score_list[int(node)]) < threshold:
            error += 1

    for node in target_list:
        if float(score_list[int(node)]) > threshold:
            target_detect += 1

    # save initial target detection
    if turn == 0:
        constants.initial_target_detected = target_detect

    # compute AUC
    if attack == "ENM":
        y_true = [0] * (num_negative - 100) + [1] * (num_positive - 100)
    else:
        y_true = [0] * (num_negative - 100) + [1] * (num_positive - 100 + constants.num_new_nodes)
    roc_auc = roc_auc_score(y_true, score_list_no_train)

    # print and save the performance of the RICC
    if constants.turn % interval == 0:
        msg = "Epoch : [{}/{}]\tFNR : [{}/{} ({:.0f}%)]\tAUC : {:.4f}".format(turn, epoch - 1, target_detect, target_num, 100. * target_detect / target_num, roc_auc)
        print(msg)
        f_performance.write("{}\n".format(msg))

    f_performance.close()
    f_score.close()
    f_target.close()
    f_train_ori.close()


def run():
    """ running function
    :return:
    """
    choose_random_train_set()
    trainset2prior()
    run_lbp(True)
    compute_diff()
    update_trainset()
    check_FN_nodes()
    run_lbp(False)
    utils.delete_record(cfg)

    constants.turn += 1


def init_var():
    # handling strategy
    type_list = ["equal_rand", "equal_lcc", "uni_rand", "uni_lcc", "cat_rand", "cat_lcc", "equal_close", "uni_close", "cat_close"]
    if types not in type_list:
        raise Exception("Wrong types.")
        print("Type should be one of {}.".format(type_list))
        exit(1)
        
    # init graph
    if attack == "ENM":
        if graph == "Enron":
            n = 67392
            constants.num_negative = 33696
            constants.num_positive = 33696

        elif graph == "Facebook":
            n = 8078
            constants.num_negative = 4039
            constants.num_positive = 4039

        elif graph == "Twitter_small":
            n = 8167
            constants.num_negative = 7358
            constants.num_positive = 809

        elif graph == "Twitter_large":
            n = 11197772 + 100000 + 10000000
            constants.num_negative = 10000000
            constants.num_positive = 100000
            constants.num_unlabel = 11197772

        else:
            raise Exception("Wrong dataset")

    elif attack == "NNI":
        if "_N_" in options:
            constants.num_new_nodes = int(options.split("_N_")[-1])

        if graph == "Enron":
            n = 67392 + constants.num_new_nodes
            constants.num_negative = 33696
            constants.num_positive = 33696

        elif graph == "Facebook":
            n = 8078 + constants.num_new_nodes
            constants.num_negative = 4039
            constants.num_positive = 4039

        elif graph == "Twitter_small":
            n = 8167 + constants.num_new_nodes
            constants.num_negative = 7358
            constants.num_positive = 809

        elif graph == "Twitter_large":
            n = 21297772 + constants.num_new_nodes
            constants.num_negative = 10000000
            constants.num_positive = 100000
            constants.num_unlabel = 11197772

        else:
            raise Exception("Wrong dataset")
    else:
        raise Exception("Wrong attack")

    # if the 'is_delete' option is TRUE, delete the case directory
    if is_delete:
        utils.delete_case(cfg)
        exit(1)

    constants.n = n
    constants.graph_list_original = [[] * n for _ in range(n)]
    constants.graph_list_attacked = [[] * n for _ in range(n)]

    SybilSCAR.read_graph(graph, types, attacked=True, attack=attack, option=options)
    SybilSCAR.read_graph(graph, types, attacked=False, attack=attack, option=options)
    utils.check_directory(cfg)


def main():
    """ main function
    :return:
    """
    # initialize variables
    init_var()

    # Run program "epoch" times
    for N in range(epoch):
        run()


if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    # Read config varaibles
    # For convenience, we use the config variable as global variables
    with open("../configs/{}".format(args.config)) as f:
        cfg = yaml.safe_load(f)

    case = "case" + cfg["case"]
    graph = cfg["graph"]
    edge_manipulation_cost = cfg["edge_manipulation_cost"]
    target_node_type = cfg["target_node_type"]
    attack = cfg["attack"]
    epoch = cfg["epoch"]
    lr = cfg["lr"]
    theta = cfg["theta"]
    weight = cfg["weight"]
    buffer = cfg["buffer"]
    sampling_size = cfg["sampling_size"]
    iteration = cfg["iteration"]
    threshold = cfg["threshold"]
    interval = cfg["interval"]
    num_modified_edges = cfg["num_modified_edges"]
    num_added_nodes = cfg["num_added_nodes"]
    num_target_nodes = cfg["num_target_nodes"]
    
    is_delete = False
    types = "{}_{}".format(edge_manipulation_cost, target_node_type)
    
    # handling option
    if attack == "ENM":
        if num_modified_edges != 30:
            options = "_K_{}".format(num_modified_edges)
        elif num_target_nodes != 100:
            options = "_T_{}".format(num_target_nodes)
        else:
            options = ""
    elif attack == "NNI":
        if num_modified_edges != 70:
            options = "_K_{}".format(num_modified_edges)
        elif num_target_nodes != 100:
            options = "_T_{}".format(num_target_nodes)
        elif num_added_nodes != 60:
            options = "_N_{}".format(num_added_nodes)
        else:
            options = ""

    # Fixed global variables
    score_path = constants.score_path
    result_path = constants.result_path
    dataset_path = constants.dataset_path

    # Run main function
    main()

