"train" and "dev" are annotated with 10 fine-grained NER categories: person, geo-location, company, facility, product, music artist, movie, sports team, tv show and other.  

Baseline(CRF) results for reference:

10 Entity Types:
accuracy:  93.68%; precision:  40.34%; recall:  32.22%; F1:  35.83
          company: precision:  43.48%; recall:  25.64%; F1:  32.26
         facility: precision:  19.44%; recall:  18.42%; F1:  18.92
          geo-loc: precision:  49.18%; recall:  51.72%; F1:  50.42
            movie: precision:  16.67%; recall:   6.67%; F1:   9.52
      musicartist: precision:   0.00%; recall:   0.00%; F1:   0.00
            other: precision:  28.57%; recall:  18.18%; F1:  22.22
           person: precision:  52.04%; recall:  59.65%; F1:  55.59
          product: precision:  12.00%; recall:   8.11%; F1:   9.68
       sportsteam: precision:  33.33%; recall:   8.57%; F1:  13.64
           tvshow: precision:   0.00%; recall:   0.00%; F1:   0.00

Note that this is just an example output for evaluating metrics. In this assignment, it is normal for the F1-score to be around 20~30.