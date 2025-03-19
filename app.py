import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random


MODEL_PATH = "./emojIQ_deepseekmath-r1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")


def emojiq_brainpower():  
    logic = random.randint(40, 100)  
    emoji_confusion = 100 - logic  
    caffeine = random.randint(10, 50)  
    return (  
        f"🧠 {logic}% logic, {emoji_confusion}% 'Wait... is 🍕 a number?!' 🤔, "  
        f"and {caffeine}% caffeine boost ☕🚀"  
    )  


st.set_page_config(page_title="EmojIQ - Emoji Math Solver", page_icon="🔢", layout="centered")
st.title("🔢 EmojIQ: Emoji Math Solver")
st.subheader("Cracking Emoji Codes, Solving Math with a Smile! 😃➕🎭")

st.write("Enter an emoji math problem, and let EmojIQ solve it! 🤓")

user_input = st.text_area("📝 Emoji Math Problem:", placeholder="e.g., 🍎 + 🍎 + 🍎 = 12")


st.sidebar.header("⚙️ Model Settings")
temperature = st.sidebar.slider("Temperature (0 = logical, 1 = creative)", 0.0, 1.5, 0.7, 0.1)
max_length = st.sidebar.slider("Max Output Length", 50, 200, 100, 10)
top_p = st.sidebar.slider("Top-p (sampling diversity)", 0.0, 1.0, 0.9, 0.05)

def solve_emoji_math(problem):
    prompt = f"Solve: {problem}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return solution.split("Answer:")[-1].strip()

if st.button("🔍 Solve Emoji Math"):
    if user_input.strip():
        with st.spinner("Thinking in emojis... 🤖🔢"):
            solved_text = solve_emoji_math(user_input)
            st.success("✅ Solution:")
            st.write(f"**{solved_text}**")
            
            st.info(emojiq_brainpower())
    else:
        st.warning("⚠️ Please enter an emoji math problem to solve.")

st.markdown("---")
st.caption("🔢 **EmojIQ** - Cracking Emoji Codes, One Equation at a Time! 🚀")