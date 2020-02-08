import numpy as np
from naive_bayes_methods import naive_bayes, write_results_to_txt, read_results_to_dict


options = {}
#options['directory'] = "G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/train"
options['directory'] = "G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/train_10"
#write_results_to_txt([],[],[])
#dict_spam_ham,dict_ham_tokens,dict_spam_tokens = read_results_to_dict(file_name = "/models/nbmodel_one_add.txt")
nb_model = naive_bayes()
nb_model.learn_nb(options['directory'])
test_directory = "G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/dev"
#test_directory = "G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/train"
nb_model.classify_nb(test_directory)
results = {}
results['accuracy'], results['percision_ham'], results['recall_ham'], results['f_score_ham'], results['percision_spam'], results['recall_spam'], results['f_score_spam'] = nb_model.evaluate_nb("nboutput.txt")
 