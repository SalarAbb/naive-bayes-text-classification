import numpy as np
from naive_bayes_methods import naive_bayes, write_results_to_txt


options = {}
options['directory'] = "G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/train"
write_results_to_txt([],[],[])
nb_model = naive_bayes()
nb_model.learn_nb(options['directory'])
test_directory = "G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/dev"
#test_directory = "G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/train"
labels_t,labels_c = nb_model.classify_nb(test_directory)
accuracy, f_score_ham, f_score_spam = nb_model.evaluate_nb(labels_t,labels_c)
a = 1
 