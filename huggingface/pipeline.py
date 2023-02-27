# Pipeliens are basically a combination of preprocessing -> model -> postprocessing

from transformers import pipeline

# Just tell what model or task you want 
classifier = pipeline("sentiment-analysis")
out = classifier("We are very happy to show you the 🤗 Transformers library.")
print(out)

generator = pipeline("text-generation")
out = generator("We are very happy to show you the 🤗 Transformers library.", max_new_tokens=20)
print(out)

ner_model = pipeline("ner", grouped_entities=True)
out = ner_model("Who is sashank gondala")
print(out)

masker_model = pipeline("fill-mask")
out = masker_model("Cat chases the <mask>.", top_k=5)
print(out)