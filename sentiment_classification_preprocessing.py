
# coding: utf-8

# In[107]:


import numpy as np
import pandas as pd


# In[108]:


#data reading and preprocessing
df_train = pd.read_csv('D:/Assignment_ML/sentiment_tweet/training_data.csv',encoding = "ISO-8859-1" )
#df_test = pd.read_csv('D:/Assignment_ML/sentiment_tweet/testdata.manual.2009.06.14.csv',encoding = "ISO-8859-1" )  
df =df_train


# In[109]:


#df =df_train.append(df_test)
df.head()


# In[110]:


df.head(5)
df.columns = ['polarity of tweet','tweet_id','date of tweet','query','user_id','tweet_text']
df.head()


# In[111]:


# data statistics
neg = len(df[df["polarity of tweet"] == 0])
pos = len(df[df["polarity of tweet"] == 4])
neu = len(df[df["polarity of tweet"] == 2])
print (neg, neu, pos, len(df))


# In[112]:


noquery = len(df[df["query"] == "NO_QUERY"])
print(noquery)


# In[113]:


df_test = pd.read_csv('D:/Assignment_ML/sentiment_tweet/testdata.manual.2009.06.14.csv',encoding = "ISO-8859-1" ) 
df_test.head()


# In[114]:


df_test.columns=['polarity of tweet','x','date of tweet','device','user_id','tweet_text']
df_test.head()


# In[115]:


df_test =df_test[['polarity of tweet','tweet_text']]
df_test.head()


# In[116]:


df_train= df[['polarity of tweet','tweet_text']]
df_train.head()


# In[117]:


# important feature to do classification target: noquery, variable :tweet_text
df =df_train.append(df_test)
df.head()


# In[118]:


# data statistics
neg = len(df[df["polarity of tweet"] == 0])
pos = len(df[df["polarity of tweet"] == 4])
neu = len(df[df["polarity of tweet"] == 2])
print (neg, neu, pos, len(df))


# In[121]:


df.to_csv('D:/Assignment_ML/sentiment_tweet/processed_training_data.csv')


# In[122]:


df['clean_tweet'] = df['tweet_text']
df.head(10)


# In[123]:


#tweet cleaning process
#Step 1 : Removing "@user" from all the tweets
#Step 2 : Changing all the tweets into lowercase
#Step 3 : Apostrophe Lookup
#Step 4 : Short Word Lookup
#Step 5 : Emoticon Lookup
#Step 6 : Replacing Special Characters with space
#Step 7 : Replacing Numbers (integers) with space
#Step 8 : Removing words whom length is 1

#Step 1 : Removing "@user" from all the tweets
import re
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


# In[124]:


df['clean_tweet'] = np.vectorize(remove_pattern)(df['clean_tweet'], "@[\w]*")


# In[125]:


df.head()


# In[126]:


#Step 2 : Changing all the tweets into lowercase
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: x.lower())
df.head(10)


# In[127]:


#Step 3 : Apostrophe Lookup
# Apostrophe Dictionary
apostrophe_dict = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
apostrophe_dict


# In[128]:


def lookup_dict(text, dictionary):
    for word in text.split():
        if word.lower() in dictionary:
            if word.lower() in text.split():
                text = text.replace(word, dictionary[word.lower()])
    return text


# In[129]:


df['clean_tweet'] = df['clean_tweet'].apply(lambda x: lookup_dict(x,apostrophe_dict))
df.head(10)


# In[130]:


#Step 4 : Short Word Lookup
short_word_dict = {
"121": "one to one",
"a/s/l": "age, sex, location",
"adn": "any day now",
"afaik": "as far as I know",
"afk": "away from keyboard",
"aight": "alright",
"alol": "actually laughing out loud",
"b4": "before",
"b4n": "bye for now",
"bak": "back at the keyboard",
"bf": "boyfriend",
"bff": "best friends forever",
"bfn": "bye for now",
"bg": "big grin",
"bta": "but then again",
"btw": "by the way",
"cid": "crying in disgrace",
"cnp": "continued in my next post",
"cp": "chat post",
"cu": "see you",
"cul": "see you later",
"cul8r": "see you later",
"cya": "bye",
"cyo": "see you online",
"dbau": "doing business as usual",
"fud": "fear, uncertainty, and doubt",
"fwiw": "for what it's worth",
"fyi": "for your information",
"g": "grin",
"g2g": "got to go",
"ga": "go ahead",
"gal": "get a life",
"gf": "girlfriend",
"gfn": "gone for now",
"gmbo": "giggling my butt off",
"gmta": "great minds think alike",
"h8": "hate",
"hagn": "have a good night",
"hdop": "help delete online predators",
"hhis": "hanging head in shame",
"iac": "in any case",
"ianal": "I am not a lawyer",
"ic": "I see",
"idk": "I don't know",
"imao": "in my arrogant opinion",
"imnsho": "in my not so humble opinion",
"imo": "in my opinion",
"iow": "in other words",
"ipn": "I’m posting naked",
"irl": "in real life",
"jk": "just kidding",
"l8r": "later",
"ld": "later, dude",
"ldr": "long distance relationship",
"llta": "lots and lots of thunderous applause",
"lmao": "laugh my ass off",
"lmirl": "let's meet in real life",
"lol": "laugh out loud",
"ltr": "longterm relationship",
"lulab": "love you like a brother",
"lulas": "love you like a sister",
"luv": "love",
"m/f": "male or female",
"m8": "mate",
"milf": "mother I would like to fuck",
"oll": "online love",
"omg": "oh my god",
"otoh": "on the other hand",
"pir": "parent in room",
"ppl": "people",
"r": "are",
"rofl": "roll on the floor laughing",
"rpg": "role playing games",
"ru": "are you",
"shid": "slaps head in disgust",
"somy": "sick of me yet",
"sot": "short of time",
"thanx": "thanks",
"thx": "thanks",
"ttyl": "talk to you later",
"u": "you",
"ur": "you are",
"uw": "you’re welcome",
"wb": "welcome back",
"wfm": "works for me",
"wibni": "wouldn't it be nice if",
"wtf": "what the fuck",
"wtg": "way to go",
"wtgp": "want to go private",
"ym": "young man",
"gr8": "great"
}
short_word_dict


# In[131]:


df['clean_tweet'] = df['clean_tweet'].apply(lambda x: lookup_dict(x,short_word_dict))
df.head(10)


# In[132]:


#Step 5 : Emoticon Lookup
emoticon_dict = {
":)": "happy",
":‑)": "happy",
":-]": "happy",
":-3": "happy",
":->": "happy",
"8-)": "happy",
":-}": "happy",
":o)": "happy",
":c)": "happy",
":^)": "happy",
"=]": "happy",
"=)": "happy",
"<3": "happy",
":-(": "sad",
":(": "sad",
":c": "sad",
":<": "sad",
":[": "sad",
">:[": "sad",
":{": "sad",
">:(": "sad",
":-c": "sad",
":-< ": "sad",
":-[": "sad",
":-||": "sad"
}
emoticon_dict


# In[133]:


df['clean_tweet'] = df['clean_tweet'].apply(lambda x: lookup_dict(x,emoticon_dict))
df.head(10)


# In[134]:


#Step 6 : Replacing Special Characters with space
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
df.head(10)


# In[135]:


#Step 7 : Replacing Numbers (integers) with space
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
df.head(10)


# In[136]:


#Step 8 : Removing words whom length is 1
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>1]))
df['clean_tweet'][0:10]


# In[138]:


df.to_csv('D:/Assignment_ML/sentiment_tweet/processed_training_data_preprocessed.csv')

