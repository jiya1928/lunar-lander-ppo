import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="LUNARIS", layout="wide")

st.markdown("<h1 style='text-align:center;'>LUNARIS - RESUME PROJECT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Train AI To Land A Rocket | Reward Shaping | LunarLander-v3</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### CONTROL PANEL")
    mode = st.selectbox("Select Agent:", ["Baseline", "Hacked - Hovering", "Fixed - Fast Landing"])
    speed = st.slider("Speed", 1, 10, 5)
    launch = st.button("LAUNCH SIMULATION")

# Hero image upar - video jaisa look
if os.path.exists("hero.png"):
    st.image("hero.png", use_container_width=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### LIVE LANDING PAD")
    video_box = st.empty()  # Yaha live video aayega
    status_box = st.empty()

with col2:
    st.markdown("### SCORE")
    score_box = st.empty()
    score_box.markdown("<h1 style='text-align:center; color:green;'>+0</h1>", unsafe_allow_html=True)
    st.markdown("### Training Progress")
    chart_box = st.empty()

# --- ASLI LIVE SIMULATION ---
if launch:
    try:
        import gymnasium as gym
        # Live env banaya - rgb_array matlab har frame ek photo banega
        env = gym.make("LunarLander-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        total_score = 0
        scores = []
        
        for step in range(300):
            # Mode ke hisaab se action
            if mode == "Hacked - Hovering":
                # Hawa me latka rahega - upar ka thruster bar-bar
                action = 2 if step % 3 == 0 else 0
            elif mode == "Fixed - Fast Landing":
                # Seedha neeche aayega
                action = 2 if obs[1] > -0.2 else 0
                if abs(obs[0]) > 0.1:
                    action = 1 if obs[0] < 0 else 3
            else:
                action = env.action_space.sample()  # Random - Baseline jaisa

            obs, reward, done, truncated, _ = env.step(action)
            
            # Fixed me har frame -0.1 penalty - yehi reward shaping hai
            if mode == "Fixed - Fast Landing":
                reward -= 0.1
            total_score += reward
            scores.append(total_score)

            # Har frame ko live dikhao - YEHI VIDEO JAISE LAGEGA
            frame = env.render()
            video_box.image(frame, use_container_width=True)
            
            color = "red" if total_score < 0 else "green"
            score_box.markdown(f"<h1 style='text-align:center; color:{color};'>SCORE<br>{int(total_score)}</h1>", unsafe_allow_html=True)
            
            # Graph bhi live update
            fig, ax = plt.subplots()
            ax.plot(scores, color='#FFAA00')
            ax.set_title("Live Score")
            chart_box.pyplot(fig)
            plt.close(fig)

            status_box.write(f"Step: {step} | Height: {round(float(obs[1]),2)} | Speed: {round(float(obs[3]),2)} | Action: {action}")
            
            time.sleep(0.05 / speed)  # Speed control
            
            if done or truncated:
                if total_score > 100:
                    status_box.success(f"LANDED! Final Score: {int(total_score)} - Safe Landing +100")
                elif total_score < 0:
                    status_box.error(f"CRASH! Final Score: {int(total_score)} - Crash -100")
                else:
                    status_box.warning(f"Finished! Final Score: {int(total_score)}")
                break
        env.close()
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Gymnasium install nahi hai? Ye chala: python -m pip install gymnasium[box2d]")
else:
    st.info("LAUNCH SIMULATION dabao - Rocket live udega, video jaisa.")

st.markdown("---")
st.markdown("**Kya Seekha?** Baseline slow hai, Hacked fake score deta hai, Fixed reward shaping se fast land karta hai.")