from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
classifier = pipeline('sentiment-analysis')
print(classifier('We are very happy to include pipeline into the transformers repository.'))
print(classifier('so bad'))
print(classifier('我很开心'))
print(classifier('我非常伤心'))