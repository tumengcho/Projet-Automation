import spacy
from spacy.training import Example
import json
import random


def train_custom_ner_model(data_path, model_output_path="ner_model"):
    # Load JSON intent
    with open(data_path, 'r') as f:
        intent = json.load(f)

    # Initialize spaCy model
    nlp = spacy.blank("en")


    # Extract unique entity labels
    labels = set()
    for example in intent['examples']:
        for entity in example['entities']:
            label = entity['entity']
            if label not in labels:
                labels.add(label)
                # Add NER component to the pipeline for the new label
                print(nlp.pipe_names)
                if "ner" in nlp.pipe_names:
                    nlp.remove_pipe("ner")
                ner = nlp.add_pipe("ner")
                ner.add_label(label)

    # Prepare training intent
    train_data = []
    for example in intent['examples']:
        text = example['text']
        entities = example['entities']
        entities_info = [(entity['start'], entity['end'], entity['entity']) for entity in entities]
        train_data.append((text, {"entities": entities_info}))

    # Convert training intent to spaCy format
    examples = []
    for text, annots in train_data:
        examples.append(Example.from_dict(nlp.make_doc(text), annots))

    # Train the NER model
    nlp.begin_training()
    for _ in range(20):  # Adjust number of epochs as needed
        random.shuffle(examples)
        losses = {}
        for example in examples:
            nlp.update([example], losses=losses)
        print("Losses:", losses)

    # Save the trained model
    nlp.to_disk(model_output_path)

def teach_and_update_model():
    # Load the trained NER model
    nlp = spacy.load("ner_model")

    # Load JSON data
    with open('Intents/test.json', 'r') as f:
        data = json.load(f)
    with open("Intents/Intent.json", 'r') as r:
        intent = json.load(r)

    # Define a function to extract application names from text
    def extract_entities(text, entity_labels):
        doc = nlp(text)
        entities_info = [(ent.label_, ent.text) for ent in doc.ents if ent.label_ in entity_labels]
        return entities_info

    # Simple chatbot loop
    print("Hello! I'm a chatbot. Type a command to launch an application.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input.lower() == "train":
            train_custom_ner_model('Intents/test.json', model_output_path="ner_model")

        else:
            labels = set()
            for example in data['examples']:
                for entity in example['entities']:
                    label = entity['entity']
                    if label not in labels:
                        labels.add(label)
            applications_info = extract_entities(user_input.lower(), labels)
            print(applications_info)
            if applications_info:
                for label, value in applications_info:
                    print(f"{value}... -- {label}")
            else:
                print("I couldn't recognize any application. Can you teach me?")

                # Prompt user to provide entity and value
                print("Existing entity: ")
                for l in labels:
                    print(l)
                entity = input("What is the entity? (e.g., APPLICATION): ").upper()  # Convert to uppercase
                value = input("What is its value? (e.g., Spotify): ").lower()  # Convert to lowercase

                # Find start index of value in lowercase input text
                start_index = user_input.lower().index(value)

                replaced = user_input.replace(value,f"[{entity}]")
                print(replaced)
                for i in intent["intents"]:
                    if(entity == i["entityType"]):
                        for t in i['text']:
                            if(t.lower() == replaced.lower()):
                                i['text'].append(replaced)
                        break
                    else:
                        print("Non existent")
                # Add new entity and value to the JSON data
                data['examples'].append({
                     "text": user_input,
                     "entities": [{
                         "entity": entity,
                         "value": value,
                         "start": start_index,
                         "end": start_index + len(value)
                     }]
                 })

                # Save the updated JSON data
                with open('Intents/test.json', 'w') as f:
                     json.dump(data, f, indent=4)
                with open('Intents/Intent.json', 'w') as r:
                    json.dump(intent, r, indent=4)
                print(f"Thank you for teaching me! I'll remember {value} next time. -- {replaced}")

                # Re-train the NER model with the updated data
                # (This part should be done offline and not in the chatbot loop)
                # You can run the training script provided earlier in this conversation

        # Call the function to start the chatbot and update the model
teach_and_update_model()
train_custom_ner_model("Intents/test.json")
