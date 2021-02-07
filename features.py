
import nltk
import re
import word_category_counter
import data_helper
import os, sys
from nltk.util import ngrams
from nltk.corpus import opinion_lexicon

DATA_DIR = "data"
LIWC_DIR = "liwc"

word_category_counter.load_dictionary(LIWC_DIR)


def normalize(token, should_normalize=True):
    """
    This function performs text normalization.

    If should_normalize is False then we return the original token unchanged.
    Otherwise, we return a normalized version of the token, or None.

    For some tokens (like stopwords) we might not want to keep the token. In
    this case we return None.

    :param token: str: the word to normalize
    :param should_normalize: bool
    :return: None or str
    """
    
    if not should_normalize:
        normalized_token = token

    else:
        token_lower = token.lower()
        stop_words = nltk.corpus.stopwords.words('english')
        if token_lower in stop_words:
            return None
        if re.findall(r'[\w]', token_lower) == []:
            return None
        normalized_token = token_lower

        
        #raise NotImplemented

    return normalized_token



def get_words_tags(text, should_normalize=True):
    """
    This function performs part of speech tagging and extracts the words
    from the review text.

    You need to :
        - tokenize the text into sentences
        - word tokenize each sentence
        - part of speech tag the words of each sentence

    Return a list containing all the words of the review and another list
    containing all the part-of-speech tags for those words.

    :param text:
    :param should_normalize:
    :return:
    """
    words = []
    tags = []

    # tokenization for each sentence

    tokens_sent = nltk.sent_tokenize(text)

    words_in_sent = [nltk.word_tokenize(sentence) for sentence in tokens_sent]
    #print(words_in_sent)
    words = [word for sentences in words_in_sent for word in sentences]
    #print(words)
    word_tags = nltk.pos_tag(words)
    tags = [tup[1] for tup in word_tags]
    #print(pos)
    #raise NotImplemented

    return words, tags


def feature_ngram(tokens, ngrams_count, feature_vectors):
    tmp_dict = {}
    grams = ngrams(tokens, ngrams_count)
    if ngrams_count == 1:
        keys = ['UNI_'+tup[0] for tup in grams]
    if ngrams_count == 2:
        keys = ['BI_' + tup[0] + "_"+tup[1] for tup in grams]
    if ngrams_count == 3:
        keys = ['TRI_' + tup[0] + "_"+tup[1] + "_"+tup[2] for tup in grams]
    token_len = len(keys)
    for key in keys:
        if key in tmp_dict:
            tmp_dict[key] += 1
        if key not in tmp_dict:
            tmp_dict[key] = 1
    #print(feature_vectors)
    for entry in tmp_dict:
        tmp_dict[entry] = round(tmp_dict[entry]/token_len,3)
    #print(tmp_dict)
    [feature_vectors.update( {tup : tmp_dict[tup]} ) for tup in tmp_dict]

    return feature_vectors
    #print(feature_vectors)

def feature_ngram_count(tokens, ngrams_count, feature_vectors):
    tmp_dict = {}
    grams = ngrams(tokens, ngrams_count)
    if ngrams_count == 1:
        keys = ['UNI_'+tup[0] for tup in grams]
    if ngrams_count == 2:
        keys = ['BI_' + tup[0] + "_"+tup[1] for tup in grams]
    if ngrams_count == 3:
        keys = ['TRI_' + tup[0] + "_"+tup[1] + "_"+tup[2] for tup in grams]
    token_len = len(keys)
    for key in keys:
        if key in tmp_dict:
            tmp_dict[key] += 1
        if key not in tmp_dict:
            tmp_dict[key] = 1
    #print(feature_vectors)
    #print(tmp_dict)
    [feature_vectors.update( {tup : tmp_dict[tup]} ) for tup in tmp_dict]
    return feature_vectors

def get_ngram_count_features(tokens):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :return: feature_vectors: a dictionary values for each ngram feature
    """
    feature_vectors = {}
    tokens_normalized = [normalize(token,True) for token in tokens ]
    tokens_normalized_final = [ token for token in tokens_normalized if token is not None]
    tokens_len_unigrams = (len(tokens_normalized_final))
    

    feature_ngram_count(tokens_normalized_final, 1, feature_vectors)
    feature_ngram_count(tokens_normalized_final, 2, feature_vectors)
    feature_ngram_count(tokens_normalized_final, 3, feature_vectors)
    #print(feature_vectors)
    

    #raise NotImplemented

    return feature_vectors

def get_ngram_features(tokens):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :return: feature_vectors: a dictionary values for each ngram feature
    """
    feature_vectors = {}
    tokens_normalized = [normalize(token,True) for token in tokens ]
    tokens_normalized_final = [ token for token in tokens_normalized if token is not None]
    tokens_len_unigrams = (len(tokens_normalized_final))
    
    ##UNIGRAMS
    '''unigrams = ngrams(tokens_normalized_final, 1)
    keys_uni = ['UNI_'+word[0] for word in unigrams]
    for key in keys_uni:
        if key in feature_vectors:
            feature_vectors[key] += 1
        if key not in feature_vectors:
            feature_vectors[key] =1
    #print(feature_vectors)
    for entry in feature_vectors:
        feature_vectors[entry] = round(feature_vectors[entry]/tokens_len_unigrams,3)

    #print(feature_vectors)
    #feature_ngram(tokens_normalized_final, 1, feature_vectors)
   
    bigrams_wf = nltk.bigrams(tokens_normalized_final)

    keys_bi = ['BI_' + tup[0] + "_"+tup[1] for tup in bigrams_wf]
    tokens_len_bigrams = (len(keys_bi))

    for key in keys_bi:
        if key in feature_vectors:
            feature_vectors[key] += 1
        if key not in feature_vectors:
            feature_vectors[key] =1
    print(feature_vectors)

    for entry in feature_vectors:
        feature_vectors[entry] = round(feature_vectors[entry]/tokens_len_bigrams,3)'''

    feature_ngram(tokens_normalized_final, 1, feature_vectors)
    feature_ngram(tokens_normalized_final, 2, feature_vectors)
    feature_ngram(tokens_normalized_final, 3, feature_vectors)
    #print(feature_vectors)
    
    #raise NotImplemented

    return feature_vectors


def get_pos_features(tags):
    """
    This function creates the unigram and bigram part-of-speech features
    as described in the assignment3 handout.

    :param tags: list of POS tags
    :return: feature_vectors: a dictionary values for each ngram-pos feature
    """
    feature_vectors = {}
    #print(tags)
    feature_ngram(tags, 1, feature_vectors)
    feature_ngram(tags, 2, feature_vectors)
    feature_ngram(tags, 3, feature_vectors)
    #print(feature_vectors)


    #raise NotImplemented

    return feature_vectors



def get_liwc_features(words):
    """
    Adds a simple LIWC derived feature

    :param words:
    :return:
    """

    # TODO: binning

    feature_vectors = {}
    text = " ".join(words)
    liwc_scores = word_category_counter.score_text(text)
    # All possible keys to the scores start on line 269
    # of the word_category_counter.py script
    negative_score = liwc_scores["Negative Emotion"]
    positive_score = liwc_scores["Positive Emotion"]
    negations = liwc_scores["Negations"]
    sad = liwc_scores["Sadness"]
    positive_feeling = liwc_scores["Positive feelings"]
    assent = liwc_scores["Assent"]
    leisure = liwc_scores['Leisure']

    ### 2 GIVEN FEATURES
    feature_vectors["Negative Emotion"] = negative_score
    feature_vectors["Positive Emotion"] = positive_score
    ### 5 ADDED FEATURES
    feature_vectors["Negations"] = negations
    feature_vectors["Sadness"] = sad
    feature_vectors["Positive feelings"] = positive_feeling
    feature_vectors["Assent"] = assent
    feature_vectors["Leisure"] = leisure


    if negations > assent:
        feature_vectors['liwc:negations'] = 1
    else:
        feature_vectors['liwc:assent'] = 1

    if sad > positive_feeling:
        feature_vectors['liwc:sadness'] = 1
    else:
        feature_vectors['liwc:positive feelings'] = 1
    
    if leisure > 0:
        feature_vectors['liwc:leisure'] = 1
    else:
         feature_vectors['liwc:leisure'] = 0

    if positive_score > negative_score:
        feature_vectors["liwc:positive"] = 1
    else:
        feature_vectors["liwc:negative"] = 1
    
    #print(feature_vectors)
    #raise NotImplemented
    return feature_vectors


FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "word_pos_opinion_features", "word_pos_stop_words_features", "word_pos_liwc_opinion_features", "word_count_features"}
 #"word_pos_stop_words_features" also tried out


def get_opinion_features(tags):
    """
    This function creates the opinion lexicon features
    as described in the assignment3 handout.

    the negative and positive data has been read into the following lists:
    * neg_opinion
    * pos_opinion

    if you haven't downloaded the opinion lexicon, run the following commands:
    *  import nltk
    *  nltk.download('opinion_lexicon')

    :param tags: tokens
    :return: feature_vectors: a dictionary values for each opinion feature
    """
    feature_vectors = {}
    neg_opinion = opinion_lexicon.negative()
    pos_opinion = opinion_lexicon.positive()
    #print(tags)
    normalize_tags = [normalize(word) for word in tags if word]
    normalized_tokens = [word for word in normalize_tags if word is not None]
    #print(normalized_tokens)
    
    '''for word in normalized_tokens:
        if word in neg_opinion or word in pos_opinion:
            #feature_vectors.update({word: 1})
            feature_vectors[word] = 1
        else:
            #feature_vectors.update({word: 0})
            feature_vectors[word] = 0'''
    for word in neg_opinion:
        if word in normalized_tokens:
            feature_vectors.update({word: 1})
        else:
            feature_vectors.update({word: 0})

    for word in pos_opinion:
        if word in normalized_tokens:
            feature_vectors.update({word: 1})
        else:
            feature_vectors.update({word: 0})

    #print(feature_vectors)
    ###     YOUR CODE GOES HERE
    #raise NotImplemented

    return feature_vectors

def get_stop_words(tokens):
    feature_vectors = {}
    stop_words = nltk.corpus.stopwords.words('english')

    lowered_words = [word.lower() for word in tokens ]

    word_final = [word for word in lowered_words if re.findall(r'[\w]', word) != []]
    #print(word_final)
    #print(lowered_words)
    for word in word_final: 
        if word in stop_words:
            feature_vectors[word] = 1
        else:
            feature_vectors[word] = 0
    #print(tokens)
    #print(feature_vectors)
    #raise NotImplemented

    return feature_vectors


def get_features_category_tuples(category_text_dict, feature_set):
    """

    You will might want to update the code here for the competition part.

    :param category_text_dict:
    :param feature_set:
    :return:
    """
    features_category_tuples = []
    all_texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)

    # FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "word_pos_opinion_features"}
    #print(feature_set)
    for category in category_text_dict:
        for text in category_text_dict[category]:

            words, tags = get_words_tags(text)
            feature_vectors = {}
            #print(words)
            #print(tags)
            if feature_set == "word_features":
                feature_vectors = get_ngram_features(words)
                #print(category)
                #file_wf = open('word_features-DATASET-features.txt','w')

            if feature_set == "word_pos_features":
                feature_vectors = get_ngram_features(words)
                feature_vectors_pos = get_pos_features(tags)
                [feature_vectors.update({entry: feature_vectors_pos[entry]}) for entry in feature_vectors_pos]
                #print(feature_vectors)

            if feature_set == "word_pos_liwc_features":
                feature_vectors = get_ngram_features(words)
                feature_vectors_pos = get_pos_features(tags)
                [feature_vectors.update({entry: feature_vectors_pos[entry]}) for entry in feature_vectors_pos]
                feature_vectors_liwc = get_liwc_features(words)
                [feature_vectors.update({entry: feature_vectors_liwc[entry]}) for entry in feature_vectors_liwc]

            ### TAKES A BIT TO RUN
            if feature_set == "word_pos_opinion_features":
                feature_vectors = get_ngram_features(words)
                feature_vectors_pos = get_pos_features(tags)
                [feature_vectors.update({entry: feature_vectors_pos[entry]}) for entry in feature_vectors_pos]
                feature_vectors_opinion = get_opinion_features(words)
                [feature_vectors.update({entry: feature_vectors_opinion[entry]}) for entry in feature_vectors_opinion]
            
            if feature_set == "word_pos_stop_words_features":
                feature_vectors = get_ngram_features(words)
                feature_vectors_pos = get_pos_features(tags)
                [feature_vectors.update({entry: feature_vectors_pos[entry]}) for entry in feature_vectors_pos]
                feature_vectors_stop = get_stop_words(words)
                [feature_vectors.update({entry: feature_vectors_stop[entry]}) for entry in feature_vectors_stop]
            
            if feature_set ==  "word_pos_liwc_opinion_features":
                feature_vectors = get_ngram_features(words)
                feature_vectors_pos = get_pos_features(tags)
                [feature_vectors.update({entry: feature_vectors_pos[entry]}) for entry in feature_vectors_pos]
                feature_vectors_liwc = get_liwc_features(words)
                [feature_vectors.update({entry: feature_vectors_liwc[entry]}) for entry in feature_vectors_liwc]
                feature_vectors_opinion = get_opinion_features(words)
                [feature_vectors.update({entry: feature_vectors_opinion[entry]}) for entry in feature_vectors_opinion]
            
            if feature_set ==  "word_count_features":
                feature_vectors = get_ngram_count_features(words)

            ###     YOUR CODE GOES HERE
            #raise NotImplemented

            features_category_tuples.append((feature_vectors, category))
            all_texts.append(text)

    return features_category_tuples, all_texts


def write_features_category(features_category_tuples, outfile_name):
    """
    Save the feature values to file.

    :param features_category_tuples:
    :param outfile_name:
    :return:
    """
    with open(outfile_name, "w", encoding="utf-8") as fout:
        for (features, category) in features_category_tuples:
            #features_list = list(features)
            #print(features_list)
            features_list_format = "\t".join([ key+ ":"+ str(features[key]) for key in features])
            #print(features_list_format)
            fout.write("{0:<10s}\t{1}\n".format(category, features_list_format))



def write_to_file(datafile, feature_set):
    
    raw_data = data_helper.read_file(datafile)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}

    features_category_tuples, texts = get_features_category_tuples(category_texts, feature_set)

    #format_dict = "\t".join([ key+ ":"+ str(tup[0][key]) for tup in features_category_tuples for key in tup[0]])
    #format_cat = [ tup[1] for tup in features_category_tuples]
    '''new_feature_cat_tups = []
    format_dict = []
    for tup in features_category_tuples:
        format_cat = tup[1]
        for key in tup[0]:
            format_dict.append(key+ ":"+ str(tup[0][key]))
        format_dict_final = "   ".join([entry for entry in format_dict])
    new_feature_cat_tups.append((format_dict_final,format_cat))'''

    #print(new_feature_cat_tups)
    #raise NotImplemented
    datafile_new = datafile[5:]

    filename = feature_set + "-"+ datafile_new + "-features.txt"

    write_features_category(features_category_tuples, filename)


def features_stub():
    ### CHANGED THE DATAFILE SOURCE FROM "imdb-training.data"
    '''datafile = "data/imdb-training.data"
    
    raw_data = data_helper.read_file(datafile)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}
    feature_set = "word_features"

    features_category_tuples, texts = get_features_category_tuples(category_texts, feature_set)
    #raise NotImplemented
    filename = "???"
    write_features_category(features_category_tuples, filename)

    datafile2 = "data/imdb-development.data"
    
    raw_data2 = data_helper.read_file(datafile2)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)'''

    write_to_file("data/imdb-training.data", "word_features")
    write_to_file("data/imdb-training.data", "word_pos_features")
    write_to_file("data/imdb-training.data", "word_pos_liwc_features")
    write_to_file("data/imdb-training.data", "word_pos_opinion_features")
    #write_to_file("data/imdb-training.data", "word_pos_stop_words_features")
    #write_to_file("data/imdb-training.data", "word_pos_liwc_opinion_features")


    write_to_file("data/imdb-development.data", "word_features")
    write_to_file("data/imdb-development.data", "word_pos_features")
    write_to_file("data/imdb-development.data", "word_pos_liwc_features")
    write_to_file("data/imdb-development.data", "word_pos_opinion_features")
    #write_to_file("data/imdb-development.data", "word_pos_stop_words_features")
    #write_to_file("data/imdb-development.data", "word_pos_liwc_opinion_features")

    write_to_file("data/imdb-testing.data", "word_features")
    write_to_file("data/imdb-testing.data", "word_pos_features")
    write_to_file("data/imdb-testing.data", "word_pos_liwc_features")
    write_to_file("data/imdb-testing.data", "word_pos_opinion_features")
    #write_to_file("data/imdb-testing.data", "word_pos_stop_words_features")
    #write_to_file("data/imdb-development.data", "word_pos_liwc_opinion_features")


if __name__ == "__main__":
    features_stub()
