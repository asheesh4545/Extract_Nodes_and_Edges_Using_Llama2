import os
from groq import Groq
import json
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")


client = Groq(
    api_key=api_key,
    
)

# the ontology and system prompt
ontology = """
{
    "labels": [
        {"Medicine": "A substance or preparation used to prevent or treat disease"},
        {"Condition": "A medical condition or disease that a medicine is used to treat"},
        {"Symptom": "A sign or indication of a medical condition"},
        {"Side effect": "An unintended effect of taking a medicine"},
        {"Allergy": "A reaction to a medicine that is not intended"},
        {"Active ingredient": "A substance that has a therapeutic effect. A medicine can contain an active ingredient"}
    ],
    "relationships": [
        { "has_symptom" : "Relationship between one Condition and  one Symptom" },
        { "treats" : "Relationship between Medicine and Condition" },
        { "has_sideeffects" : "Relationship between Medicine and Side effect" },
    ]
}
"""




extraction_prompt = (
    "You are an expert at creating Knowledge Graphs. "
    "Consider the following ontology. \n"
    f"{ontology} \n"
    "The user will provide you with an input text delimited by ```. "
    "Extract all the entities and relationships from the user-provided text as per the given ontology. Do not use any previous knowledge about the context. "
    "Ensure that each entity is extracted individually and not as a combined group. Each entity (such as a symptom, condition, or medicine) should be treated separately. "
    "If a text contains a list of symptoms or conditions, extract each symptom or condition as a separate entity with its respective relationships. "
    "Be consistent with the given ontology. Use ONLY the labels and relationships mentioned in the ontology. "
    "Format your output as a JSON with the following schema. \n"
    "[\n"
    "   {\n"
    '       "node_1": {"label": "as per the ontology", "name": "Name of the entity"},\n'
    '       "node_2": {"label": "as per the ontology", "name": "Name of the entity"},\n'
    '       "relationship": "Describe the relationship between node_1 and node_2 as per the context, in one to two words."\n'
    "   },\n"
    "]\n"
    "Do not add any other comment before or after the JSON. Respond ONLY with a well-formed JSON that can be directly read by a program."
)

validation_prompt = (
    "Now, validate the extracted entities and relationships. "
    "Ensure each relationship from node_1 to node_2 is sensible and logical based on the ontology. "
    "Return only the relationships that make sense in the following format: \n"
    "[\n"
    "   {\n"
    '       "node_1": {"label": "as per the ontology", "name": "Name of the entity"},\n'
    '       "node_2": {"label": "as per the ontology", "name": "Name of the entity"},\n'
    '       "relationship": "Describe the relationship between node_1 and node_2 as per the context, in one to two words."\n'
    "   },\n"
    "]\n"
    "Do not add any other comment before or after the JSON. Respond ONLY with a well-formed JSON that can be directly read by a program."
)

input_text = """
AB Phylline SR 200 Tablet can help prevent the onset of an asthma attack if it is taken before exercise or exposure to some “triggers”. These may include house dust, pollen, pets and cigarette smoke. This medicine will allow you to exercise more freely without worrying about getting symptoms such as wheezing, coughing and shortness of breath. It allows you to live your life more freely without worrying so much about things that set off your symptoms.
"""


extraction_response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": extraction_prompt + f"```{input_text}```"
        }
    ],
    model="llama3-70b-8192", 
)


extraction_content = extraction_response.choices[0].message.content


extracted_data = json.loads(extraction_content)
validation_response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": validation_prompt + json.dumps(extracted_data, indent=4)
        }
    ],
    model="llama3-70b-8192",temperature = 0, seed = 42
)


validated_content = validation_response.choices[0].message.content
validated_data = json.loads(validated_content)
print(json.dumps(validated_data, indent=4))
