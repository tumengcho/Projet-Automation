import spacy
import json

def teach_and_update_model():
    # Load the trained NER model
    nlp = spacy.load("ner_model")

    # Load JSON data
    with open('Intents/test.json', 'r') as f:
        data = json.load(f)

    # Define a function to extract application names from text
    def extract_application(text):
        doc = nlp(text)
        applications_info = [(ent.label_, ent.text) for ent in doc.ents if ent.label_ == "APPLICATION"]
        return applications_info

    # Simple chatbot loop
    print("Hello! I'm a chatbot. Type a command to launch an application.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        else:
            applications_info = extract_application(user_input.lower())
            if applications_info:
                for label, value in applications_info:
                    print(f"Launching {value}...")
            else:
                print("I couldn't recognize any application. Can you teach me?")

                # Prompt user to provide entity and value
                entity = input("What is the entity? (e.g., APPLICATION): ").upper()  # Convert to uppercase
                value = input("What is its value? (e.g., Spotify): ").lower()  # Convert to lowercase

                # Find start index of value in lowercase input text
                start_index = user_input.lower().index(value)

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
                    json.dump(data, f)

                print(f"Thank you for teaching me! I'll remember {value} next time.")

                # Re-train the NER model with the updated data
                # (This part should be done offline and not in the chatbot loop)
                # You can run the training script provided earlier in this conversation

# Call the function to start the chatbot and update the model
teach_and_update_model()
