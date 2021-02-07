
import re, nltk, pickle, argparse
import os
import data_helper
from features import get_features_category_tuples
from contextlib import redirect_stdout

DATA_DIR = "data"


def write_features_category(features_category_tuples, output_file_name):
    output_file = open("{}-features.txt".format(output_file_name), "w", encoding="utf-8")
    for (features, category) in features_category_tuples:
        output_file.write("{0:<10s}\t{1}\n".format(category, features))
    output_file.close()


def get_classifier(classifier_fname):
    classifier_file = open(classifier_fname, 'rb')
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier


def save_classifier(classifier, classifier_fname, feature_set, data_set):
    data_correct = data_set[5:]
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()
    info_file = open(feature_set+ "-" + data_correct + '-informative-features.txt', 'w', encoding="utf-8")
    for feature, n in classifier.most_informative_features():
        info_file.write("{0}\n".format(feature))
    info_file.close()


def evaluate(classifier, features_category_tuples, reference_text, data_set_name=None):

    #print(data_set_name)
    #f = open('classifier.pickle' ,'r')
    #classifier = pickle.load(f)
    #f.close()
    features_only = [tup[0] for tup in features_category_tuples]
    reference_labels = [tup[1] for tup in features_category_tuples]
    #print(reference_labels)
    #test_file = open("test.txt", "w")
    #test_file.write(str(features_only))
    predicted_labels = classifier.classify_many(features_only)
    #print(predicted_labels)
    confusion_matrix = nltk.ConfusionMatrix(reference_labels, predicted_labels)
    #print(confusion_matrix)

    probability = []
    for pdist in classifier.prob_classify_many(features_only):
        probability.append('%.4f %.4f' % (pdist.prob('positive'), pdist.prob('negative')))
    
    #print(reference_labels)

    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)
    #print(accuracy)
    # TODO: evaluate your model
    
    #raise NotImplemented


    return accuracy, probability, confusion_matrix


def build_features(data_file, feat_name, save_feats=None, binning=False):
    # read text data
    raw_data = data_helper.read_file(data_file)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}

    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name)

    # save features to file
    if save_feats is not None:
        write_features_category(features_category_tuples, save_feats)

    return features_category_tuples, texts

def train_model(datafile, feature_set, data_set, save_model=None):

    features_data, texts = build_features(datafile, feature_set)

    # TODO: train your model here
    training_instances = features_data
    #print(training_instances)
    classifier = nltk.classify.NaiveBayesClassifier.train(training_instances)
    #f = open( "imdb-" + feature_set+"-model-P1"+ '.pickle', 'wb')
    #pickle.dump(classifier, f)
    #f.close()
    '''def save_classifier(classifier, classifier_fname):
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()
    info_file = open(classifier_fname.split(".")[0] + '-informative-features.txt', 'w', encoding="utf-8")
    for feature, n in classifier.most_informative_features(100):
        info_file.write("{0}\n".format(feature))
    info_file.close()'''
    save_classifier(classifier, "imdb-" + feature_set+"-model-P1.pickle", feature_set, data_set)
    #imdb-word pos liwc features-model

    #raise NotImplemented
    

    if save_model is not None:
        save_classifier(classifier, save_model)
    return classifier

def train_eval(train_file, feature_set, eval_file=None):

    # train the model
    split_name = "train"
    ### deleted binning = binning
    model = train_model(train_file, feature_set, eval_file)
    #model.show_most_informative_features(20)
    #model.show_most_informative_features()
    new_eval_file = eval_file[5:]
    '''with open(feature_set+"-" +new_eval_file +'-informative-features.txt', 'w') as f:
            with redirect_stdout(f):
                f.write(str(model.show_most_informative_features(100)))'''


    # save the model
    if model is None:
        model = get_classifier(classifier_fname)

    # evaluate the model
    if eval_file is not None:
        features_data, texts = build_features(eval_file, feature_set, binning=None)
        #### CHANGED data_set_name = None to data_set_name = feature_set
        accuracy, probability, cm = evaluate(model, features_data, texts, data_set_name=feature_set)
        if feature_set == "word_features":
            f_output("output-ngrams.txt", accuracy, new_eval_file, probability, cm)
        if feature_set == "word_pos_features":
            f_output("output-pos.txt", accuracy, new_eval_file, probability, cm)
        if feature_set == "word_pos_liwc_features":
            f_output("output-liwc.txt", accuracy, new_eval_file, probability, cm)
        if feature_set == "word_pos_opinion_features":
            f_output("output-opinion.txt", accuracy, new_eval_file, probability, cm)
        if feature_set == "word_pos_stop_words_features":
            f_output("output-combo-all.txt", accuracy, new_eval_file, probability, cm)
        if feature_set == "word_pos_liwc_opinion_features":
            f_output("output-combo-all.txt", accuracy, new_eval_file, probability, cm)
        if feature_set == "word_count_features":
            f_output("output-bin.txt", accuracy, new_eval_file, probability, cm)
    else:
        accuracy = None

    return accuracy

def f_output(filename, accuracy, eval_file , probability, cm):
        file_output = open(filename, 'a+')
        file_output.write("The accuracy of {} is: {}\n".format(eval_file, accuracy))
        file_output.write("Proabability per class:\n")
        file_output.write("\n")
        file_output.write(str(probability))
        file_output.write("\n")
        file_output.write("Confusion Matrix:\n")
        file_output.write(str(cm))
        file_output.write("\n")

def execute_train(default_file):
    # add the necessary arguments to the argument parser
    parser = argparse.ArgumentParser(description='Assignment 3')
    ### CHANGED deafult from "imdb-training.data"
    parser.add_argument('-d', dest="data_fname", default=default_file,
                        help='File name of the testing data.')
    args = parser.parse_args()


    train_data = args.data_fname
    #print(train_data)
    ### CHANGED deafult from "imdb-training.data"
    eval_data = "data/imdb-development.data"
    eval_data2 = "data/imdb-training.data"
    eval_data3 = "data/imdb-testing.data"


    for feat_set in ["word_features", "word_pos_features", "word_pos_liwc_features", "word_pos_opinion_features", "word_pos_stop_words_features", "word_pos_liwc_opinion_features", "word_count_features" ]:
        #used word_pos_stop_words, word_pos_liwc_opinion_features as a feature
        print("\nTraining with {}".format(feat_set))
        #acc = train_eval(train_data, feat_set, eval_file = eval_data)
        #acc = train_eval(train_data, feat_set, eval_file = eval_data2)
        acc = train_eval(train_data, feat_set, eval_file = eval_data3)

def main():
    if os.path.exists("output-ngrams.txt"):  
        output_file = open("output-ngrams.txt","w")
        output_file.write("")
        output_file.close()
    if os.path.exists("output-pos.txt"):  
        output_file = open("output-pos.txt","w")
        output_file.write("")
        output_file.close()
    if os.path.exists("output-liwc.txt"):  
        output_file = open("output-liwc.txt","w")
        output_file.write("")
        output_file.close()
    if os.path.exists("output-opinion.txt"):  
        output_file = open("output-opinion.txt","w")
        output_file.write("")
        output_file.close()

    execute_train("data/imdb-training.data")
    #execute_train("data/imdb-testing.data")
    #execute_train("data/imdb-testing.data")

    #execute_train("data/imdb-development.data")
    #execute_train("data/imdb-testing.data")
    




if __name__ == "__main__":
    main()




