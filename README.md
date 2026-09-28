## Dataset selection

Shabd is a Database for Hindi. Corpus size: 1.4 billion, ~34k manually selected words, a list of ~96k words with a freq of more than 100.
It also has a List 2.3M words and a List 2.5M most frequent bigrams.
I have downloaded it from [link](https://osf.io/xfbhd/files).

The dataset is present in /Shabd/fygme-osfstorage-archive/.

## Choosing The model

| Model | Hindi-only or multilingual | Type | Notes |
|---|---|---|---|
| `flax-community/roberta-hindi` | Hindi-only | Contextual (RoBERTa) | Trained solely on Hindi text (mC4-hi, OSCAR-hi, Hindi Wikipedia, IndicGLUE, Samanantar, Hindi news corpora) |
| `facebook/fasttext-hi-vectors` | Hindi-only (per-language model) | Static (fastText) | Trained using datasets composed of a mixture of Wikipedia and Common Crawl, one model per language, not cross-lingually aligned |
| IndicFT (AI4Bharat) | Hindi-only (per-language model) | Static (fastText) | Trained on IndicNLP Corpora, well-suited for Indian languages due to their agglutinative morphology; outperforms generic fastText on Hindi word similarity (0.598 vs 0.551) |
| `AbhishekBiswas12/word2vec-hindi` | Hindi-only | Static (Skip-gram) | A hobby/learning project — "primarily a learning and research-oriented project", trained on only ~82M tokens with a custom PyTorch pipeline (not a standard gensim/word2vec format). I'd avoid this for real research — small corpus, non-standard artifact, unclear stability. |
| `ai4bharat/indic-bert` | Multilingual (12 Indic langs jointly) | Contextual (ALBERT) | Pretrained exclusively on 12 major Indian languages: Assamese, Bengali, English, Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu, shared vocabulary/parameters across languages |