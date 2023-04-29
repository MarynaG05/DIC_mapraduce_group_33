import json
import random
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol
import re

class TextClassifier(MRJob):
    
    # Define regex pattern to split text into tokens
    WORD_RE = re.compile(r'\b\w+\b')
    
    # Load stopwords
    import nltk

   # Download stopwords
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    #nltk.download()
    stopwords = set(stopwords.words('english'))
    # Define MRJob steps
    def steps(self):
        return [
            MRStep(mapper=self.mapper_tokenize,
                combiner=self.combiner_count_tokens,
                reducer=self.reducer_count_tokens),
            
            MRStep(mapper=self.mapper_tf,
                combiner=self.combiner_sum_tf,
                reducer=self.reducer_sum_tf),
            
            MRStep(mapper=self.mapper_df,
                combiner=self.combiner_sum_df,
                reducer=self.reducer_sum_df)
        ]

    
    def reducer_compute_chi(self, term, cat_freq_dicts):
        # Calculate chi-square value for each term in each category
        # Order the terms according to their value per category
        # Preserve the top 75 terms per category
        cat_total_dict = {}
        term_total = 0
        for cat_freq_dict in cat_freq_dicts:
            for category, freq in cat_freq_dict.items():
                if category in cat_total_dict:
                    cat_total_dict[category] += freq
                else:
                    cat_total_dict[category] = freq
                term_total += freq
        N = sum(cat_total_dict.values())
        top_chi = []
        for category, cat_total in cat_total_dict.items():
            for term_cat_freq_dict in cat_freq_dicts:
                if category in term_cat_freq_dict:
                    A = term_cat_freq_dict[category]
                else:
                    A = 0
                B = cat_total - A
                C = term_total - A
                D = N - A - B - C
                chi = (N * ((A * D) - (B * C)) ** 2) / ((A + C) * (B + D) * (A + B) * (C + D))
                top_chi.append((term[1], category, chi))
        top_chi = sorted(top_chi, key=lambda x: x[2], reverse=True)[:75]

        yield term[0], top_chi   

    
    # Step 1: Tokenize text and emit unigrams as key-value pairs
           
    def mapper_tokenize(self, _, line):
    
        obj = json.loads(line)
        category = obj['category']
        reviewText = obj['reviewText']
        
        for token in self.WORD_RE.findall(reviewText):
            if token not in self.stopwords and len(token) > 1:
                yield (category, token.lower()), 1
    
        
       
    
    def combiner_count_tokens(self, key, values):
        yield key, sum(values)
    
    def reducer_count_tokens(self, key, values):
        yield key, sum(values)
    
    # Step 2: Calculate term frequencies per category
    def mapper_tf(self, key, value):
        category, term = key
        yield category, (term, value)
    
    def combiner_sum_tf(self, category, term_freqs):
        term_freq_dict = {}
        for term, freq in term_freqs:
            term_freq_dict[term] = term_freq_dict.get(term, 0) + freq
        for term, freq in term_freq_dict.items():
            yield category, (term, freq)
    
    def reducer_sum_tf(self, category, term_freqs):
        
        term_freq_dict = {}
        for term, freq in term_freqs:
            term_freq_dict[term] = term_freq_dict.get(term, 0) + freq
        yield category, term_freq_dict
    
    # Step 3: Calculate document frequencies per term per category
    def mapper_df(self, category, term_freqs):
        for term, freq in term_freqs.items():
            yield term, (category, freq)
    
    def combiner_sum_df(self, term, cat_freqs):
        cat_freq_dict = {}
        for category, freq in cat_freqs:
            cat_freq_dict[category] = cat_freq_dict.get(category, 0) + freq
        for category, freq in cat_freq_dict.items():
            yield term, (category, freq)
    
    def reducer_sum_df(self, term, cat_freqs):
            cat_freq_dict = {}
            for category, freq in cat_freqs:
                cat_freq_dict[category] = freq

            yield term, cat_freq_dict


            

           
if __name__ == '__main__':
    TextClassifier.run()