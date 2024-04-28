import spacy
from spacy.training import Example
import json
import random

def train_custom_ner_model(data_path, model_output_path="ner_model"):
    # Load JSON data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Initialize spaCy model
    nlp = spacy.blank("en")


    # Extract unique entity labels
    labels = set()
    for example in data['examples']:
        for entity in example['entities']:
            label = entity['entity']
            if label not in labels:
                labels.add(label)
                # Add NER component to the pipeline for the new label
                print(nlp.pipe_names)
                if "ner" not in nlp.pipe_names:
                    ner = nlp.add_pipe("ner")
                    if label not in ner.labels:
                        ner.add_label(label)

                else:
                    ner = nlp.get_pipe("ner")
                    if label not in ner.labels:
                        ner.add_label(label)



    # Prepare training data
    train_data = []
    for example in data['examples']:
        text = example['text']
        entities = example['entities']
        entities_info = [(entity['start'], entity['end'], entity['entity']) for entity in entities]
        train_data.append((text, {"entities": entities_info}))

    # Convert training data to spaCy format
    examples = []
    for text, annots in train_data:
        examples.append(Example.from_dict(nlp.make_doc(text), annots))

    # Train the NER model
    nlp.begin_training()
    for _ in range(100):  # Adjust number of epochs as needed
        random.shuffle(examples)
        losses = {}
        for example in examples:
            nlp.update([example], losses=losses)
        print("Losses:", losses)

    # Save the trained model
    nlp.to_disk(model_output_path)

# Example usage:
train_custom_ner_model('Intents/test.json', model_output_path="ner_model")
