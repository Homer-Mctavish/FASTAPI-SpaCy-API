import spacy
from spacy.pipeline import EntityRuler
from spacy.language import Language
import json

def create_entity_ruler(nlp, name):
    # Create a new EntityRuler
    ruler = EntityRuler(nlp, name=name)

    # Load training data from JSONL file
    with open("skill_patterns.jsonl", "r") as file:
        for line in file:
            data = json.loads(line)
            text = data["pattern"]
            label = data["label"]
            # Create pattern dictionary
            pattern = {"label": label, "pattern": text}
            # Add pattern to the EntityRuler
            ruler.add_patterns([pattern])

    return ruler

Language.factory("ent_rule", func=create_entity_ruler)


# Load existing spaCy model from disk
nlp = spacy.load("en_core_web_sm")



# Add the EntityRuler to the pipeline and specify a name
nlp.add_pipe('ent_rule', name="entity_ruler2", before="ner")

# Save the updated spaCy model
nlp.to_disk("./ERoutput")
