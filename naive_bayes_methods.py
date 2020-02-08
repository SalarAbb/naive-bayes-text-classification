import numpy as np
import os
import glob
import pickle

class naive_bayes(object):

    def __init__(self):



        pass

    def learn_nb(self,directory_to_learn):
        #
        type_smoothing = 'one_add'
        #
        list_of_ham_dirs = [x[0] for x in os.walk(directory_to_learn) if "ham" in x[0]]
        list_of_spam_dirs = [x[0] for x in os.walk(directory_to_learn) if "spam" in x[0]]
        # 1 prepare dictionary of tokens
        dict_ham_tokens = {}
        dict_spam_tokens = {}
        dict_spam_ham = {}
        dict_spam_ham['spam'] = 0
        dict_spam_ham['ham'] = 0
        # 2 define the dictionary of probabilities
        # ham
        for d in list_of_ham_dirs:
            list_txt_files = glob.glob('{}\*.txt'.format(d))         
            print('directory {} starts'.format(d))
            for file_name in list_txt_files:
                print('file {} starts'.format(file_name))
                add_to_dict_from_file(file_name,dict_ham_tokens)   
                dict_spam_ham['ham'] = dict_spam_ham['ham'] + 1
        
        print('ham is done')
        
        # spam
        for d in list_of_spam_dirs:
            print('directory {} starts'.format(d))
            list_txt_files = glob.glob('{}\*.txt'.format(d))          
            for file_name in list_txt_files:
                print('file {} starts'.format(file_name))
                add_to_dict_from_file(file_name,dict_spam_tokens)   
                dict_spam_ham['spam'] = dict_spam_ham['spam'] + 1
        print('spam is done')
        #
        dict_vocab = {**dict_spam_tokens, **dict_ham_tokens}
        # vocab_size 
        vocab_size = len(dict_vocab.keys())
        # turn dict to 
        dict_ham_tokens = turn_dict_tokens_to_prob(dict_ham_tokens,type_smoothing,vocab_size=vocab_size)
        dict_spam_tokens = turn_dict_tokens_to_prob(dict_spam_tokens,type_smoothing,vocab_size=vocab_size)

        num_files = dict_spam_ham['spam'] + dict_spam_ham['ham']         
        dict_spam_ham['spam'] = np.log(dict_spam_ham['spam'] / num_files)
        dict_spam_ham['ham'] = np.log(dict_spam_ham['ham'] / num_files)
        # 3 generate the text from dictionary
        current_directory = os.getcwd()
        folder_save = '{}/models'.format(current_directory)
        if not os.path.exists(folder_save):
                os.makedirs(folder_save)
        file_name = 'model_training_{}'.format(type_smoothing)
        file_save_path = '{}/{}'.format(folder_save,file_name)
        pickle.dump([dict_ham_tokens,dict_spam_tokens,dict_spam_ham], open(file_save_path, 'wb'))
        write_results_to_txt(dict_spam_ham,dict_ham_tokens,dict_spam_tokens,file_name="nbmodel.txt")
        pass

    def classify_nb(self,test_directory):
        type_smoothing = 'one_add'
        file_name = 'G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/models/model_training_{}'.format(type_smoothing)
        loaded_model = pickle.load(open(file_name, 'rb'))
        dict_ham_tokens,dict_spam_tokens,dict_spam_ham = loaded_model
        a = 1
        # read the text and create the dictionary
        labels_t = []
        labels_c = []
        for path, subdirs, files in os.walk(test_directory):
            for name in files:
                file_name_this = os.path.join(path, name)
                if "spam" in file_name_this or "ham" in file_name_this:
                    if "spam" in file_name_this:
                        label_this = "spam"
                    elif "ham" in file_name_this:
                        label_this = "ham"   
                    label_classified_this = self.classify_from_file_name(file_name_this,dict_ham_tokens,dict_spam_tokens,dict_spam_ham)

                    labels_t.append(label_this)
                    labels_c.append(label_classified_this)
        # classify the test from the 

        return labels_t,labels_c 
        
    def classify_from_file_name(self,file_name,dict_ham_tokens,dict_spam_tokens,dict_spam_ham):
        prob_spam = dict_spam_ham['spam']
        prob_ham = dict_spam_ham['ham']
        with open(file_name, "r", encoding="latin1") as f: 
            for line in f:
                words = line.split()
                for word in words:
                    if word.lower() in dict_ham_tokens and word.lower() in dict_spam_tokens:
                        prob_spam = prob_spam + dict_spam_tokens[word.lower()] # assumption: probabilities are log(prob)
                        prob_ham = prob_ham + dict_ham_tokens[word.lower()]
        # bayes
        if prob_spam > prob_ham:
            return "spam"
        else:
            return "ham"                    

    def evaluate_nb(self,labels_c,labels_t):
        # ham:1, spam:0
        list_c = [1 if x=='ham' else 0 for x in labels_c]
        list_t = [1 if x=='ham' else 0 for x in labels_t]
        n = len(list_c)
        list_match = [1 if list_c[i] == list_t[i] else 0 for i in range(n)]
        accuracy = np.mean(list_match)

        # get the percision
        list_correct_ham = [1 if (list_c[i]==1 and list_match[i]==1) else 0 for i in range(n)]
        list_correct_spam = [1 if (list_c[i]==0 and list_match[i]==1) else 0 for i in range(n)]

        percision_ham = sum(list_correct_ham) / sum(list_c)
        percision_spam = sum(list_correct_spam) / (n - sum(list_c))
        # get the recall
        recall_ham = sum(list_correct_ham) / sum(list_t)
        recall_spam = sum(list_correct_spam) / (n - sum(list_t))
        # build evaluation metrics
        f_score_ham = (2 * percision_ham * recall_ham) / (percision_ham + recall_ham)
        f_score_spam = (2 * percision_spam * recall_spam) / (percision_spam + recall_spam)
        return accuracy, f_score_ham, f_score_spam

def add_to_dict_from_file(file_name,dict_tokens):

    with open(file_name, "r", encoding="latin1") as f: 
        for line in f:
            words = line.split()
            for word in words:
                if word.lower() in dict_tokens:
                    dict_tokens[word.lower()] = dict_tokens[word.lower()] + 1
                else:
                    dict_tokens[word.lower()] = 1
    

    return dict_tokens 

def turn_dict_tokens_to_prob(dict_tokens,type_smoothing,vocab_size=0):

    total_num_tokens = sum(dict_tokens.values())    
    if type_smoothing == 'regular': # no smoothing
        for key,value in dict_tokens.items():
            dict_tokens[key] = np.log(value /total_num_tokens)

    elif type_smoothing == 'one_add':    
        for key,value in dict_tokens.items():
            dict_tokens[key] = np.log((value + 1)/(total_num_tokens + vocab_size))
    

    return dict_tokens                                 

def write_results_to_txt(dict_spam_ham,dict_ham_tokens,dict_spam_tokens,file_name="nbmodel.txt"):
    type_smoothing = 'one_add'
    file_name = 'G:/My Drive/Salar/USC/Courses/Term X/CSCI 544/homeworks/HW1/models/model_training_{}'.format(type_smoothing)
    loaded_model = pickle.load(open(file_name, 'rb'))
    dict_ham_tokens,dict_spam_tokens,dict_spam_ham = loaded_model
    # this function writes the results of learned spam and ham into a text file called nbmodel.txt
    file_txt = open(file_name,"w", encoding="utf-8") 
    # write Spam and Ham
    file_txt.write("ham_prob" + " " + str(dict_spam_ham['ham']) + "\n")
    file_txt.write("spam_prob" + " " + str(dict_spam_ham['spam']) + "\n") 
    file_txt.write("spam_dict" + "\n")
    for token in dict_spam_tokens:
        file_txt.write(token + " " + str(dict_spam_tokens[token]) + "\n")

    file_txt.write("ham_dict" + "\n") 
    for token in dict_ham_tokens:
        file_txt.write(token + " " + str(dict_ham_tokens[token]) + "\n")

    file_txt.close() #to change file access modes 
    return