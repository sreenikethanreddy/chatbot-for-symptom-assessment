import random
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load a pre-trained language model for natural language understanding
model_name = "gpt2"  # You can replace this with a medical-specific model if available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Expanded symptom database
symptoms = {
    "fever": ["rest", "hydrate", "take over-the-counter fever reducers", "monitor temperature"],
    "headache": ["rest", "stay in a quiet, dark room", "take over-the-counter pain relievers", "apply cold or warm compress"],
    "cough": ["stay hydrated", "use honey or cough drops", "avoid irritants", "use a humidifier"],
    "sore throat": ["gargle with salt water", "drink warm liquids", "use throat lozenges", "rest your voice"],
    "nausea": ["eat bland foods", "stay hydrated", "avoid strong odors", "try ginger tea"],
    "fatigue": ["get adequate sleep", "maintain a balanced diet", "engage in light exercise", "manage stress"],
    "diarrhea": ["stay hydrated", "eat bland foods", "avoid dairy and fatty foods", "consider probiotics"],
    "constipation": ["increase fiber intake", "stay hydrated", "exercise regularly", "try a gentle laxative"],
    "muscle pain": ["rest the affected area", "apply ice or heat", "gentle stretching", "take over-the-counter pain relievers"],
    "skin rash": ["avoid scratching", "apply cool compress", "use unscented moisturizer", "try over-the-counter antihistamines"],
    "anxiety": ["practice deep breathing", "engage in regular exercise", "try meditation", "consider talking to a therapist"],
    "insomnia": ["establish a regular sleep schedule", "create a relaxing bedtime routine", "avoid screens before bed", "limit caffeine intake"]
}

# Initialize effectiveness scores for each remedy
effectiveness_scores = {symptom: {remedy: 1.0 for remedy in remedies} for symptom, remedies in symptoms.items()}

def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def assess_symptoms(user_input):
    user_input = user_input.lower()
    for symptom, remedies in symptoms.items():
        if symptom in user_input:
            # Sort remedies by their effectiveness score
            sorted_remedies = sorted(remedies, key=lambda x: effectiveness_scores[symptom][x], reverse=True)
            top_remedies = sorted_remedies[:3]  # Recommend top 3 remedies
            return f"For {symptom}, you can try: {', '.join(top_remedies)}. If symptoms persist or worsen, please consult a healthcare professional.", symptom, top_remedies
    return "I couldn't identify a specific symptom. Could you please provide more details about how you're feeling?", None, None

def update_effectiveness(symptom, remedies, feedback):
    learning_rate = 0.1
    for remedy in remedies:
        if feedback == 'helpful':
            effectiveness_scores[symptom][remedy] += learning_rate
        elif feedback == 'not helpful':
            effectiveness_scores[symptom][remedy] -= learning_rate
        effectiveness_scores[symptom][remedy] = max(0.1, min(effectiveness_scores[symptom][remedy], 2.0))  # Keep scores between 0.1 and 2.0

def chatbot():
    print("Welcome to the Symptom Assessment Chatbot. Please note that this is not a replacement for professional medical advice.")
    print("How can I help you today?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Take care and don't hesitate to seek professional medical help if needed. Goodbye!")
            break

        # Generate a response using the language model
        ai_response = generate_response(user_input)

        # Assess symptoms and provide advice
        symptom_advice, symptom, remedies = assess_symptoms(user_input)

        # Combine AI response with symptom assessment
        response = f"{ai_response}\n\n{symptom_advice}"

        print(f"Chatbot: {response}")
        print("\nRemember, this is not a substitute for professional medical advice. If you're experiencing severe symptoms or are unsure, please consult a healthcare provider.")

        if symptom and remedies:
            feedback = input("Was this advice helpful? (yes/no): ").lower()
            if feedback in ['yes', 'y']:
                update_effectiveness(symptom, remedies, 'helpful')
                print("Thank you for your feedback. I'm glad the advice was helpful.")
            elif feedback in ['no', 'n']:
                update_effectiveness(symptom, remedies, 'not helpful')
                print("I'm sorry the advice wasn't helpful. I'll use your feedback to improve future recommendations.")

if __name__ == "__main__":
    chatbot()