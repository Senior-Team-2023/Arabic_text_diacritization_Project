# Define Diacritics
KASRA = "\u0650"
DAMMA = "\u064F"
FATHA = "\u064E"
FATHA2 = '`'
KASRATAN = "\u064D"
DAMMATAN = "\u064C"
FATHATAN = "\u064B"
SUKUN = "\u0652"
SHADDA = "\u0651"
SHADDA_DAMMA =  DAMMA + SHADDA 
SHADDA_FATHA =  FATHA + SHADDA 
SHADDA_KASRA =  KASRA + SHADDA 
SHADDA_DAMMATAN =  DAMMATAN + SHADDA
SHADDA_FATHATAN =  FATHATAN + SHADDA 
SHADDA_KASRATAN =  KASRATAN + SHADDA  

# Define a list of diacritics
# Number of Classes = diacritics + "" (No diacritic)
DIACRITICS = [DAMMA, FATHA, FATHA2,  KASRA, DAMMATAN, FATHATAN, KASRATAN, SHADDA_DAMMA, SHADDA_FATHA,  SHADDA_KASRA, SHADDA_DAMMATAN, SHADDA_FATHATAN, SHADDA_KASRATAN, SHADDA, SUKUN, ""]
ARABIC_ALPHABIT = "اآإءبتثجحخدذرزسشصضطظعغفقكلمنهويأئؤىة"

# This function is responsible for mapping diacritics to their corresponding strings
def diacritic_to_str(diacritic):
    if diacritic == SHADDA:
        diacritic = "SHADDA"
    elif diacritic == KASRA:
        diacritic = "KASRA"
    elif diacritic == DAMMA:
        diacritic = "DAMMA"
    elif diacritic == FATHA:
        diacritic = "FATHA"
    elif diacritic == FATHA2:
        diacritic = "FATHA"
    elif diacritic == KASRATAN:
        diacritic = "KASRATAN"
    elif diacritic == DAMMATAN:
        diacritic = "DAMMATAN"
    elif diacritic == FATHATAN:
        diacritic = "FATHATAN"
    elif diacritic == SUKUN:
        diacritic = "SUKUN"
    elif diacritic == SHADDA_DAMMA:
        diacritic = "SHADDA_DAMMA"
    elif diacritic == SHADDA_FATHA:
        diacritic = "SHADDA_FATHA"
    elif diacritic == SHADDA_KASRA:
        diacritic = "SHADDA_KASRA"
    elif diacritic == SHADDA_DAMMATAN:
        diacritic = "SHADDA_DAMMATAN"
    elif diacritic == SHADDA_FATHATAN:
        diacritic = "SHADDA_FATHATAN"
    elif diacritic == SHADDA_KASRATAN:
        diacritic = "SHADDA_KASRATAN"
    elif diacritic == SHADDA:
        diacritic = "SHADDA"
    else:
        diacritic = " "
    return diacritic

# This function is responsible for printing the mapping between Arabic text and diacritics
# For result debbugging purposes
def print_text_to_diacritic_mapping(text_without_diacritics, diacritic_list, is_sentence = False):
    j = 0
    for i in range(len(text_without_diacritics)):
        if is_sentence == True and (text_without_diacritics[i] == " " or text_without_diacritics[i] == '\n' or text_without_diacritics[i] == ""):
            print(text_without_diacritics[i] ,":", "")
        else:
            print(text_without_diacritics[i] ,":", diacritic_to_str(diacritic_list[j]))
            j+=1

# This function is responsible for mapping diacritics to their corresponding strings
# For result visualization purposes 
# Example :
# Input : ["", "", ّ, ِ]
# Output : ["", "", "SHADDA", "KASRA"]
def map_text_to_diacritic(diacritic_list):
    mapped_list = []
    for i in range(len(diacritic_list)):
        mapped_list.append(diacritic_to_str(diacritic_list[i]))
    return mapped_list

# This function is responsible for separating diacritics from Arabic text for the hole TRAINING data
def PrepareData(data):
    text_without_diacritics = []
    diacritic_list = []
    for i, word in enumerate(data):
        # if word == "" or word.isspace(): continue
        text, diacritic = remove_diacritics(word)
        # print(i," - ", word , " - ", text, " : ", map_text_to_diacritic(diacritic))
        text_without_diacritics.append(text)
        diacritic_list.append(diacritic)
    return text_without_diacritics, diacritic_list


# This function is responsible for separating diacritics from Arabic text for the hole TRAINING data
def RemoveDiacriticFromSentence(data):
    text_without_diacritics = []
    for i, sentence in enumerate(data):
        if sentence == "" or sentence.isspace(): continue
        text, _ = remove_diacritics(sentence,True)
        text_without_diacritics.append(text)
    return text_without_diacritics

# This function returns one hot encoding for a given arabic character
def CharToOneHOt(char):
    # Check if the given character is in the Arabic alphabet
    if char not in ARABIC_ALPHABIT:
        raise ValueError("Invalid Arabic character : \'" + char + "\'")

    # Create a dictionary mapping each character to its one-hot encoding
    one_hot_encoding = [1 if c == char else 0 for c in ARABIC_ALPHABIT]

    return one_hot_encoding


# This function is responsible for separating diacritics from Arabic text
# It returns the text without diacritics and a list of the corresponding diacritics
# Example :
# Input : "اللّهِ"
# Output : "الله" , ["", "", SHADDA, KASRA]      
def remove_diacritics(text, is_sentence = False):
    diacritic_list = []
    text_without_diacritics = ""
    if not is_sentence:
        text = " " + text + " " # Add padding 
    for i in range(len(text)):
        c = text[i]
        
        if not is_sentence and (c == " " or c == '\n' or c == ""):
            continue
        # Check if the character is a diacritic
        if c in DIACRITICS:
            # Detect ... SHADDA
            if i + 1 < len(text) and text[i+1] == SHADDA:
                if c == DAMMA:
                    diacritic_list.append(SHADDA_DAMMA)
                elif c == FATHA:
                    diacritic_list.append(SHADDA_FATHA)
                elif c == KASRA:
                    diacritic_list.append(SHADDA_KASRA)
                elif c == DAMMATAN:
                    diacritic_list.append(SHADDA_DAMMATAN)
                elif c == KASRATAN:
                    diacritic_list.append(SHADDA_KASRATAN)
                elif c == FATHATAN:
                    diacritic_list.append(SHADDA_FATHATAN)
            # Detect SHADDA ...
            elif i + 1 < len(text) and c == SHADDA:
                if text[i+1] == DAMMA:
                    diacritic_list.append(SHADDA_DAMMA)
                elif text[i+1] == FATHA:
                    diacritic_list.append(SHADDA_FATHA)
                elif text[i+1] == KASRA:
                    diacritic_list.append(SHADDA_KASRA)
                elif text[i+1] == DAMMATAN:
                    diacritic_list.append(SHADDA_DAMMATAN)
                elif text[i+1] == KASRATAN:
                    diacritic_list.append(SHADDA_KASRATAN)
                elif text[i+1] == FATHATAN:
                    diacritic_list.append(SHADDA_FATHATAN)
                else: diacritic_list.append(SHADDA)
            elif text[i-1] != SHADDA:
                diacritic_list.append(c)
        else: 
            # Check if the next character is a diacritic or shadda -> because of SHADDA ... and ... SHADDA
            if (i+1 < len(text) and text[i+1] not in DIACRITICS) or (i+1 == len(text)):
                # Add an empty string to the diacritic list
                diacritic_list.append("")
                
            text_without_diacritics += c

    # print_text_to_diacritic_mapping(text_without_diacritics,diacritic_list)
            
    assert (len(text_without_diacritics) == len(diacritic_list)), f"Error : Text : {text} Len ({len(text_without_diacritics)}) != Len diacritic ({len(diacritic_list)}) list have different sizes, "

    return text_without_diacritics, diacritic_list



    
    
#################### Test the function with an example word ####################
# # word = "نَقَلَ بَعْضُهُمْ أَنَّ الْقُهُسْتَانِيَّ كَتَبَ عَلَى هَامِشِ نُسْخَتِهِ أَنَّ هَذَا مُخْتَصٌّ بِالْأَذَانِ"
# # word = "عَنْ سَعِيدِ بْنِ الْمُسَيِّبِ"
# # word = "بُرًّا"
# word = " يَأْخُذُونَ بَعْضَ مَا تَيَسَّرَ لَهُمْ أَخْذُهُ فَيَخْتَلِسُونَهُ وَيَجْعَلُونَهُ تَحْتَهُمْ حَتَّى إذَا رَجَعُوا إلَى بُيُوتِهِمْ أَخْرَجُوهُ     "
# # word = "بُيُوتِهِمْ"
# # # for i,c in enumerate(word):
# # #     print(i,"- ",c)
# text_without_diacritics, diacritic_list = remove_diacritics(word)
# print_text_to_diacritic_mapping(text_without_diacritics,diacritic_list)
# print("Text without diacritics : ", text_without_diacritics)
s = diacritic_to_str("ِ")
print(diacritic_to_str(s))