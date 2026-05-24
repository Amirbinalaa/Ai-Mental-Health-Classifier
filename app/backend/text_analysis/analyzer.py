import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
import re

# Download necessary NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

class TextAnalyzer:
    """Class for analyzing text inputs for mental health indicators using NLP and VADER"""

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.depression_markers = [
            r"\b(sad|depress(ed|ion|ing)?|hopeless|worthless|tired|exhaust(ed)?|empty|crying|cry|grief|lonely|lone|sadness)\b",
            r"\b(no(\s+)point|no(\s+)purpose|meaningless|useless|worthless)\b",
            r"\b(can't(\s+)sleep|sleeping(\s+)too(\s+)much|insomnia|sleepless)\b",
            r"\b(don't(\s+)enjoy|lost(\s+)interest|nothing(\s+)matters|care(\s+)anymore|don't(\s+)care)\b"
        ]
        self.anxiety_markers = [
            r"\b(anxious|anxiety|nervous|worried|worry|worrying|scared|afraid|panic|dread|trembl(ing|e)|shak(ing|e))\b",
            r"\b(heart(\s+)racing|sweating|shaking|chest(\s+)pain|palpitations)\b",
            r"\b(what(\s+)if|overthinking|obsess(ing|ive)?|can't(\s+)focus|ruminat(ing|e))\b",
            r"\b(overwhelmed|too(\s+)much|can't(\s+)handle|stressed(\s+)out|tension|tense)\b"
        ]
        self.stress_markers = [
            r"\b(pressure|deadlines?|overwork(ed)?|burnout|exhaust(ed|ion)?|stress(ed|ful)?)\b",
            r"\b(no(\s+)time|busy|hectic|chaotic|rushed|hours?)\b",
            r"\b(tension|irritab(le|ility)|frustrat(ed|ing|ion)|angry|mad)\b"
        ]
        self.severe_risk_markers = [
            r"\b(kill(\s+)myself|suicide|end(\s+)my(\s+)life|want(\s+)to(\s+)die|dying|wanna(\s+)die)\b",
            r"\b(self(\s+)harm|cutting(\s+)myself|end(\s+)it(\s+)all|better(\s+)off(\s+)dead|no(\s+)reason(\s+)to(\s+)live)\b",
            r"\b(hanging(\s+)myself|slit(\s+)my(\s+)wrists|take(\s+)my(\s+)own(\s+)life)\b"
        ]
        self.stop_words = {
            'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now'
        }

    def analyze(self, text):
        if not text or len(text.strip()) == 0:
            return {"error": "Empty text provided"}
        raw_words = word_tokenize(text)
        sentences = sent_tokenize(text)
        clean_words = [word.lower() for word in raw_words if word.isalnum()]
        word_count = len(clean_words)
        sentence_count = len(sentences)
        avg_word_length = sum(len(word) for word in clean_words) / word_count if word_count > 0 else 0
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        depression_count = sum(len(re.findall(p, text.lower())) for p in self.depression_markers)
        anxiety_count = sum(len(re.findall(p, text.lower())) for p in self.anxiety_markers)
        stress_count = sum(len(re.findall(p, text.lower())) for p in self.stress_markers)
        severe_risk_count = sum(len(re.findall(p, text.lower())) for p in self.severe_risk_markers)
        if compound_score >= 0.35:
            primary_emotion = "joy"
        elif compound_score <= -0.15:
            if severe_risk_count > 0:
                primary_emotion = "crisis"
            elif anxiety_count > depression_count and anxiety_count >= stress_count:
                primary_emotion = "anxiety"
            elif stress_count > depression_count and stress_count >= anxiety_count:
                primary_emotion = "stress"
            elif depression_count > 0:
                primary_emotion = "sadness"
            else:
                primary_emotion = "anger"
        else:
            primary_emotion = "neutral"
        emotions = [{"label": primary_emotion, "score": round(abs(compound_score), 3)}]
        unique_words = set(word.lower() for word in clean_words)
        linguistic_diversity = len(unique_words) / word_count if word_count > 0 else 0
        first_person_pronouns = len([word for word in clean_words if word in {'i','me','my','mine','myself'}])
        first_person_ratio = first_person_pronouns / word_count if word_count > 0 else 0
        key_phrases = []
        for i in range(len(clean_words)-1):
            w1, w2 = clean_words[i], clean_words[i+1]
            if (w1 not in self.stop_words or w2 not in self.stop_words) and w1.isalpha() and w2.isalpha():
                key_phrases.append(f"{w1} {w2}")
        for i in range(len(clean_words)-2):
            w1, w2, w3 = clean_words[i], clean_words[i+1], clean_words[i+2]
            if (w1 not in self.stop_words or w2 not in self.stop_words or w3 not in self.stop_words) and w1.isalpha() and w2.isalpha() and w3.isalpha():
                key_phrases.append(f"{w1} {w2} {w3}")
        seen = set()
        unique_key_phrases = []
        for phrase in key_phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique_key_phrases.append(phrase)
        return {
            "metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_word_length": round(avg_word_length, 2),
                "linguistic_diversity": round(linguistic_diversity, 3),
                "first_person_ratio": round(first_person_ratio, 3)
            },
            "sentiment": {
                "compound": round(compound_score, 3),
                "pos": round(sentiment_scores["pos"], 3),
                "neu": round(sentiment_scores["neu"], 3),
                "neg": round(sentiment_scores["neg"], 3)
            },
            "emotion": emotions[0],
            "markers": {
                "depression": depression_count,
                "anxiety": anxiety_count,
                "stress": stress_count,
                "severe_risk": severe_risk_count
            },
            "key_phrases": unique_key_phrases[:8]
        }
