import os
from groq import Groq
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


ONTOLOGY = {
    "labels": [
        "Product",
        "Ingredient",
        "Property",
        "TherapeuticProperty",
        "TargetCondition",
        "TechnologyFeature",
        "BeneficialEffect"
    ],
    "relationships": [
        "contains",
        "treatsOrManages",
        "hasProperty",
        "utilizes",
        "enhances",
        "alleviates" ]}


# Define extraction prompt as a constant
EXTRACTION_PROMPT = f"""
You are an expert at creating Knowledge Graphs. 
Consider the following ontology:
{json.dumps(ONTOLOGY, indent=2)}

The user will provide you with an input text delimited by ```. 
Extract all the entities and relationships from the user-provided text as per the given ontology. Use your knowledge to understand and apply the labels and relationships appropriately.
Ensure that each entity is extracted individually and not as a combined group. Each entity should be treated separately. 
If a text contains a list of symptoms, conditions, or other entities, extract each one as a separate entity with its respective relationships. 
Be consistent with the given ontology. Use ONLY the labels and relationships mentioned in the ontology. 
Format your output as a JSON with the following schema:
[
   {{
       "node_1": {{"label": "as per the ontology", "name": "Name of the entity"}},
       "node_2": {{"label": "as per the ontology", "name": "Name of the entity"}},
       "relationship": "Relationship between node_1 and node_2 as per the ontology"
   }},
]
Do not add any other comment before or after the JSON. Respond ONLY with a well-formed JSON that can be directly read by a program.
"""

def extract_knowledge_graph(input_text):
    """Extract knowledge graph from input text using Groq API."""
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{EXTRACTION_PROMPT}```{input_text}```"
            }
        ],
        model="llama3-70b-8192",
    )
    
    response_content = chat_completion.choices[0].message.content
    return json.loads(response_content)

def main():
    #input_text = """
    #AB Phylline SR 200 Tablet is used to treat and prevent symptoms of asthma, bronchitis, and chronic obstructive pulmonary disorder (a lung disorder in which flow of air to the lung is blocked). It helps in relaxing the muscles of the air passages, thus widening it and making it easier to breathe.
    #"""

    # input_text = """
    # Glycomet 500 SR Tablet is a medicine used to treat type 2 diabetes mellitus. It helps control blood sugar levels and thus prevent serious complications of diabetes. It is also used to treat a menstruation-related disorder known as Polycystic ovary syndrome (PCOS) in women.Glycomet 500 SR Tablet is best taken with food to avoid nausea and abdominal pain. You should take it regularly, at the same time each day, to get the most benefit. You should not stop taking this medicine unless your doctor recommends it. Your lifestyle plays a big part in controlling diabetes. Therefore, it is important to stay on the diet and exercise program recommended by your doctor while taking this medicine. 
    # """

    # input_text = """ 
    # Pulmoclear Tablet is used to treat and prevent asthma and symptoms of chronic obstructive pulmonary disorder (a lung disorder in which the flow of air to the lungs is blocked) such as coughing, wheezing, and shortness of breath. It helps relax the muscles of the air passages thereby making it easier to breathe.
    # Pulmoclear Tablet is taken with or without food in a dose and duration as advised by the doctor. The dose you are given will depend on your condition and how you respond to the medicine. You should keep taking this medicine for as long as your doctor recommends. If you stop treatment too early your symptoms may come back and your condition may worsen. Let your healthcare team know about all other medications you are taking as some may affect, or be affected by this medicine.
    # """

    # input_text = """ Liveril U 300mg Tablet is used to dissolve certain gallstones and prevent them from forming. It is also used to treat a type of liver disease called primary biliary cirrhosis. It helps break down the cholesterol that has converted into stones in your gallbladder thereby dissolving the stones.
    # Liveril U 300mg Tablet should be swallowed whole after a meal and with a glass of milk or water. The dose will depend on what you are being treated for and your body weight. Take it regularly to get maximum benefit and keep taking it for as long as prescribed (several months or longer). Keep taking it even if your symptoms disappear.
    # This medicine's most common side effects are abdominal pain, diarrhea, hair loss, itching, nausea, and rash. Not everyone gets these side effects. If you are worried about them, or they do not go away, let your doctor know.
    # """

    input_text = """ 
    Zeroharm Sciences Holo Oncolis Tablet is a precision nutraceutical product comprising natural extracts of curcumin, peptide, saponins, bacosides and piperine. With its antioxidant, anti-inflammatory, anti-cancer and anti-depression properties- holo oncolis can offer holistic cancer care and alleviate the associated stress and anxiety. The patent-pending encapsulation method, formulation process and nanotechnology can ensure solubility and bioavailability of ingredients, thus offering higher efficacy and therapeutic action even with a much lower dosage.
    """

    extracted_data = extract_knowledge_graph(input_text)
    print(json.dumps(extracted_data, indent=4))

if __name__ == "__main__":
    main()