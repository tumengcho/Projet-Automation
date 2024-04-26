import spacy
from spacy.training import Example
import json
import random

def train_custom_ner_model(data_path, label="APPLICATION", model_output_path="ner_model"):
    # Load JSON data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Initialize spaCy model
    nlp = spacy.blank("en")

    # Add NER component to the pipeline
    ner = nlp.add_pipe("ner")

    # Define entity labels
    ner.add_label(label)

    # Prepare training data
    train_data = []
    for example in data['examples']:
        text = example['text']
        entities = example['entities']
        entities_info = [(entity['start'], entity['end'], label) for entity in entities]
        train_data.append((text, {"entities": entities_info}))

    # Convert training data to spaCy format
    examples = []
    for text, annots in train_data:
        examples.append(Example.from_dict(nlp.make_doc(text), annots))

    # Train the NER model
    nlp.begin_training()
    for _ in range(20):  # Adjust number of epochs as needed
        random.shuffle(examples)
        for example in examples:
            nlp.update([example], losses={})

    # Save the trained model
    nlp.to_disk(model_output_path)

# Example usage:
train_custom_ner_model('Intents/test.json', label="APPLICATION", model_output_path="ner_model")


