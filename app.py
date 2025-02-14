import streamlit as st
from transformers import pipeline

# Cache the model loading for efficiency
@st.cache_resource
def load_model():
    return pipeline('text-generation', model='gpt2')

generator = load_model()

# Function to generate a recipe
def generate_recipe(ingredients, meal_type, cuisine):
    prompt = f"""Create a {cuisine} {meal_type} recipe using {ingredients}. Provide:
- Ingredients
- Instructions
- Serving Size
- Notes"""
    
    try:
        recipe = generator(
            prompt,
            max_length=500,  # Reduced length to prevent hallucinations
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )[0]['generated_text']
        
        return recipe.replace(prompt, "").strip()
    except Exception as e:
        return f"Error generating recipe: {e}"

# Streamlit UI
def main():
    st.title("🍽️ Leftover Food Recipe Generator")
    st.write("Enter your ingredients, meal type, and cuisine preference to generate a recipe!")
    
    ingredients = st.text_area("List your ingredients (one per line)")
    meal_type = st.selectbox("Meal Type:", ["Dinner", "Lunch", "Breakfast", "Snack"])
    cuisine = st.selectbox("Cuisine:", ["Italian", "Mexican", "Indian", "Chinese", "American"])
    
    if st.button("Generate Recipe"):
        if ingredients.strip():
            with st.spinner("Generating recipe..."):
                recipe = generate_recipe(ingredients, meal_type, cuisine)
                
                # Display formatted output
                st.subheader("Generated Recipe:")
                st.markdown(f"### Ingredients:\n{ingredients}\n\n### Recipe:\n{recipe}")
        else:
            st.warning("Please enter at least one ingredient.")

if __name__ == "__main__":
    main()
