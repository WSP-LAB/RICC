import os
import yaml
import shutil

import constants
import SybilSCAR


def check_directory(cfg):
    """ check the input case's environment and handle it
    :return:
    """
    # call global variables
    turn = constants.turn
    score_path = constants.score_path
    result_path = constants.result_path
    dataset_path = constants.dataset_path

    # Read config varaibles
    case = "case" + cfg["case"]
    graph = cfg["graph"]
    edge_manipulation_cost = cfg["edge_manipulation_cost"]
    target_node_type = cfg["target_node_type"]
    attack = cfg["attack"]
    theta = cfg["theta"]
    weight = cfg["weight"]
    buffer = cfg["buffer"]
    iteration = cfg["iteration"]
    num_modified_edges = cfg["num_modified_edges"]
    num_added_nodes = cfg["num_added_nodes"]
    num_target_nodes = cfg["num_target_nodes"]
    
    options = ""
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
        elif num_added_nodes != 100:
            options = "_T_{}".format(num_added_nodes)
        else:
            options = ""

    # check result directory has appropriate directory
    if not os.path.isdir(score_path + "/{}/{}".format(graph, types)):
        print("Initializing result directory ...")
        os.makedirs(result_path + "/performance/{}/{}".format(graph, types))
        os.makedirs(result_path + "/score/{}/{}".format(graph, types))
        os.makedirs(result_path + "/trainset/{}/{}".format(graph, types))
        os.makedirs(score_path + "/{}/{}/posterior".format(graph, types))
        os.makedirs(score_path + "/{}/{}/prior".format(graph, types))
        os.makedirs(score_path + "/{}/{}/diff".format(graph, types))

    # If the first try with input case
    # make directory with the input case
    if not os.path.isdir(score_path + "/{}/{}/posterior/{}".format(graph, types, case)):
        print("Creating {} directories ...".format(case))
        os.makedirs(score_path + "/{}/{}/posterior/{}".format(graph, types, case))
        os.makedirs(score_path + "/{}/{}/prior/{}".format(graph, types, case))
        os.makedirs(score_path + "/{}/{}/diff/{}".format(graph, types, case))
        os.makedirs(result_path + "/trainset/{}/{}/{}".format(graph, types, case))
        os.makedirs(result_path + "/performance/{}/{}/{}".format(graph, types, case))

        # initialize with required initial files
        post_path = score_path + "/{}/{}/posterior/{}/post_sybilscar_0.txt".format(graph, types, case)
        SybilSCAR.init(graph, 0, post_path, iteration, theta, weight * buffer, is_train=True, attacked=True)

        shutil.copyfile(dataset_path + "/initial_files/{}/{}/train_0.txt".format(graph, types),
                        result_path + "/trainset/{}/{}/{}/train_0.txt".format(graph, types, case))
        if attack == "NNI":
            if "_N_" in options:
                _option = options
            else:
                _option = ""
            shutil.copyfile(dataset_path + "/initial_files/{}/{}/prior_0({})NNI{}.txt".format(graph, types, theta, _option),
                            score_path + "/{}/{}/prior/{}/prior_0.txt".format(graph, types, case))
            shutil.copyfile(dataset_path + "/initial_files/{}/{}/prior_0({})NNI{}.txt".format(graph, types, theta, _option),
                            score_path + "/{}/{}/prior/{}/prior_0_evaluation.txt".format(graph, types, case))
        else:
            shutil.copyfile(dataset_path + "/initial_files/{}/{}/prior_0({}).txt".format(graph, types, theta),
                            score_path + "/{}/{}/prior/{}/prior_0.txt".format(graph, types, case))
            shutil.copyfile(dataset_path + "/initial_files/{}/{}/prior_0({}).txt".format(graph, types, theta),
                            score_path + "/{}/{}/prior/{}/prior_0_evaluation.txt".format(graph, types, case))

        post_path = score_path + "/{}/{}/posterior/{}/post_sybilscar_0_evaluation.txt".format(graph, types, case)
        SybilSCAR.init(graph, 0, post_path, iteration, theta, weight, is_train=True, attacked=True)

        # make post_before_attack file
        train_path = dataset_path + "/graph/{}/train.txt".format(graph)
        post_path = dataset_path + "/initial_files/{}/{}/post_sybilscar_before_attack({}_{}).txt".format(graph, types, weight, iteration)
        SybilSCAR.init(graph, 0, post_path, iteration, theta, weight, is_train=True, attacked=False)

        print("done.\n")

    # If input case directory was already existed
    # delete existing data and restart
    else:
        print("{} directory is already exist. Do you want to delete existing data and restart? [Y/n]".format(case))
        restart = input()

        # when user wants to quit
        if restart == "n":
            exit(1)
        # when user wants to delete existing data and restart.
        else:
            print("Delete existing {} data ...".format(case))
            # delete existing data
            shutil.rmtree(score_path + "/{}/{}/posterior/{}".format(graph, types, case))
            shutil.rmtree(score_path + "/{}/{}/prior/{}".format(graph, types, case))
            shutil.rmtree(score_path + "/{}/{}/diff/{}".format(graph, types, case))
            shutil.rmtree(result_path + "/trainset/{}/{}/{}".format(graph, types, case))
            shutil.rmtree(result_path + "/performance/{}/{}/{}".format(graph, types, case))

            # Re-create case directories
            os.makedirs(score_path + "/{}/{}/posterior/{}".format(graph, types, case))
            os.makedirs(score_path + "/{}/{}/prior/{}".format(graph, types, case))
            os.makedirs(score_path + "/{}/{}/diff/{}".format(graph, types, case))
            os.makedirs(result_path + "/trainset/{}/{}/{}".format(graph, types, case))
            os.makedirs(result_path + "/performance/{}/{}/{}".format(graph, types, case))

            # initialize with required initial files
            train_path = dataset_path + "/graph/{}/train.txt".format(graph)
            post_path = score_path + "/{}/{}/posterior/{}/post_sybilscar_0.txt".format(graph, types, case)
            SybilSCAR.init(graph, train_path, post_path, iteration, theta, weight * buffer, is_train=True, attacked=True)

            shutil.copyfile(dataset_path + "/initial_files/{}/{}/train_0.txt".format(graph, types),
                            result_path + "/trainset/{}/{}/{}/train_0.txt".format(graph, types, case))

            if attack == "NNI":
                if "_N_" in options:
                    _option = options
                else:
                    _option = ""
                shutil.copyfile(dataset_path + "/initial_files/{}/{}/prior_0({})NNI{}.txt".format(graph, types, theta, _option),
                                score_path + "/{}/{}/prior/{}/prior_0.txt".format(graph, types, case))
                shutil.copyfile(dataset_path + "/initial_files/{}/{}/prior_0({})NNI{}.txt".format(graph, types, theta, _option),
                                score_path + "/{}/{}/prior/{}/prior_0_evaluation.txt".format(graph, types, case))
            else:
                shutil.copyfile(dataset_path + "/initial_files/{}/{}/prior_0({}).txt".format(graph, types, theta),
                                score_path + "/{}/{}/prior/{}/prior_0.txt".format(graph, types, case))
                shutil.copyfile(dataset_path + "/initial_files/{}/{}/prior_0({}).txt".format(graph, types, theta),
                                score_path + "/{}/{}/prior/{}/prior_0_evaluation.txt".format(graph, types, case))

            train_path = dataset_path + "/graph/{}/train.txt".format(graph)
            post_path = score_path + "/{}/{}/posterior/{}/post_sybilscar_0_evaluation.txt".format(graph, types, case)
            SybilSCAR.init(graph, train_path, post_path, iteration, theta, weight, is_train=True, attacked=True)

            # make post_before_attack file
            train_path = dataset_path + "/graph/{}/train.txt".format(graph)
            post_path = dataset_path + "/initial_files/{}/{}/post_sybilscar_before_attack({}_{}).txt".format(graph, types, weight, iteration)
            SybilSCAR.init(graph, train_path, post_path, iteration, theta, weight, is_train=True, attacked=False)

            print("done.\n")


def delete_case(cfg):
    """ If you don't need data of specific case anymore, this function clean the data of specific case
    except performance data
    :return:
    """
    print("Deleting data of {} ...".format(case))

    # call global variables
    score_path = constants.score_path
    result_path = constants.result_path

    # Read config varaibles
    case = "case" + cfg["case"]
    graph = cfg["graph"]
    edge_manipulation_cost = cfg["edge_manipulation_cost"]
    target_node_type = cfg["target_node_type"]
    types = "{}_{}".format(edge_manipulation_cost, target_node_type)

    if os.path.isdir(score_path + "/{}/{}/posterior/{}".format(graph, types, case)):
        shutil.rmtree(score_path + "/{}/{}/posterior/{}".format(graph, types, case))
    if os.path.isdir(score_path + "/{}/{}/prior/{}".format(graph, types, case)):
        shutil.rmtree(score_path + "/{}/{}/prior/{}".format(graph, types, case))
    if os.path.isdir(score_path + "/{}/{}/diff/{}".format(graph, types, case)):
        shutil.rmtree(score_path + "/{}/{}/diff/{}".format(graph, types, case))
    if os.path.isdir(result_path + "/trainset/{}/{}/{}".format(graph, types, case)):
        shutil.rmtree(result_path + "/trainset/{}/{}/{}".format(graph, types, case))
    if os.path.isdir(result_path + "/performance/{}/{}/{}".format(graph, types, case)):
        shutil.rmtree(result_path + "/performance/{}/{}/{}".format(graph, types, case))
    print("done.")


def delete_record(cfg):
    """ If you don't need want to record all data in each iteration,
    this function clean the data of most of the iteration
    except performance
    :return:
    """
    # call global variables
    turn = constants.turn
    score_path = constants.score_path
    result_path = constants.result_path

    # Read config varaibles
    case = "case" + cfg["case"]
    graph = cfg["graph"]
    edge_manipulation_cost = cfg["edge_manipulation_cost"]
    target_node_type = cfg["target_node_type"]
    types = "{}_{}".format(edge_manipulation_cost, target_node_type)
    interval = cfg["interval"]

    if constants.turn % interval != 2:
        if os.path.isfile(score_path + "/{}/{}/posterior/{}/post_sybilscar_{}_evaluation.txt".format(graph, types, case, turn - 2)):
            os.remove(score_path + "/{}/{}/posterior/{}/post_sybilscar_{}_evaluation.txt".format(graph, types, case, turn - 2))
        if os.path.isfile(score_path + "/{}/{}/posterior/{}/post_sybilscar_{}.txt".format(graph, types, case, turn - 2)):
            os.remove(score_path + "/{}/{}/posterior/{}/post_sybilscar_{}.txt".format(graph, types, case, turn - 2))
        if os.path.isfile(score_path + "/{}/{}/posterior/{}/post_sybilscar_{}_prime.txt".format(graph, types, case, turn - 2)):
            os.remove(score_path + "/{}/{}/posterior/{}/post_sybilscar_{}_prime.txt".format(graph, types, case, turn - 2))

        if os.path.isfile(score_path + "/{}/{}/prior/{}/prior_{}_evaluation.txt".format(graph, types, case, turn - 2)):
            os.remove(score_path + "/{}/{}/prior/{}/prior_{}_evaluation.txt".format(graph, types, case, turn - 2))
        if os.path.isfile(score_path + "/{}/{}/prior/{}/prior_{}_prime.txt".format(graph, types, case, turn - 2)):
            os.remove(score_path + "/{}/{}/prior/{}/prior_{}_prime.txt".format(graph, types, case, turn - 2))
        if os.path.isfile(score_path + "/{}/{}/prior/{}/prior_{}.txt".format(graph, types, case, turn - 2)):
            os.remove(score_path + "/{}/{}/prior/{}/prior_{}.txt".format(graph, types, case, turn - 2))

        if os.path.isfile(score_path + "/{}/{}/diff/{}/diff_{}_sorted.txt".format(graph, types, case, turn - 2)):
            os.remove(score_path + "/{}/{}/diff/{}/diff_{}_sorted.txt".format(graph, types, case, turn - 2))
        if os.path.isfile(score_path + "/{}/{}/diff/{}/diff_{}.txt".format(graph, types, case, turn - 2)):
            os.remove(score_path + "/{}/{}/diff/{}/diff_{}.txt".format(graph, types, case, turn - 2))

        if os.path.isfile(result_path + "/trainset/{}/{}/{}/train_{}_prime.txt".format(graph, types, case, turn - 2)):
            os.remove(result_path + "/trainset/{}/{}/{}/train_{}_prime.txt".format(graph, types, case, turn - 2))

        if os.path.isfile(result_path + "/performance/{}/{}/{}/post_sybilscar_after_defense_{}.txt".format(graph, types, case, turn - 2)):
            os.remove(result_path + "/performance/{}/{}/{}/post_sybilscar_after_defense_{}.txt".format(graph, types, case, turn - 2))


def indicator(n):
    """ just for print order
    :param n: order
    :return: appropriate string for the order
    """
    if n > 9:
        return str(n)+("th" if 4 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))
    else:
        return str(n) + (" th" if 4 <= n % 100 <= 20 else {1: " st", 2: " nd", 3: " rd"}.get(n % 10, "th"))

